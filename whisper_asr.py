import torch
import numpy as np
import time
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import Config
from post_processor import PostProcessor
from audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class WhisperASR:
    def __init__(self):
        self.device = Config.DEVICE
        self.whisper_model = None
        self.processor = None
        self.post_processor = PostProcessor()
        self.audio_processor = AudioProcessor()
        self.model_loaded = False
        self.context_cache = {}
        
    def load_models(self):
        """简化模型加载"""
        if self.model_loaded:
            return
            
        try:
            logger.info("🔄 加载 Whisper 模型...")
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"📊 加载前显存: {initial_memory:.1f}G")
            
            self.processor = WhisperProcessor.from_pretrained(Config.MODEL_PATH)
            
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                Config.MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.whisper_model.eval()
            
            for param in self.whisper_model.parameters():
                param.requires_grad = False
                
            self.model_loaded = True
            
            if self.device.type == "cuda":
                after_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ 模型加载完成 - 显存占用: {after_memory:.1f}G")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            raise

    def transcribe_audio(self, audio_bytes: bytes, forced_language: str = None, 
                    session_id: str = None, use_context: bool = True) -> tuple:
        """主转录接口 - 使用 Silero VAD 分割 + 信任VAD机制"""
        if not self.model_loaded:
            return "模型未加载", "error"
            
        try:
            start_time = time.time()
            
            # 1. 基础音频预处理
            logger.info("🎯 步骤1: 音频预处理")
            audio_np = self.audio_processor.preprocess_audio(audio_bytes)
            audio_duration = len(audio_np) / 16000
            logger.info(f"预处理完成: {len(audio_np)}样本, {audio_duration:.2f}秒")
            
            # 2. 音频验证（基础检查）
            if not self.audio_processor.validate_audio(audio_np):
                logger.warning("音频质量验证失败，返回空结果")
                return "未检测到有效语音", "auto"
            
            # 3. 获取上下文提示词（如果启用）
            prompt_text = "粤语客服对话"
            if use_context and session_id and session_id in self.context_cache:
                prompt_text = self.context_cache[session_id]
                logger.info(f"📝 使用上下文提示词: {prompt_text[:100]}...")
            
            # 4. 使用 VAD 进行语音分割
            logger.info("🎯 步骤2: VAD 语音活动检测")
            speech_segments = self.audio_processor.vad_segmentation(audio_np)
            
            if not speech_segments:
                logger.warning("VAD 未检测到语音片段")
                return "未检测到语音", "auto"
            
            logger.info(f"✅ VAD 检测到 {len(speech_segments)} 个语音片段")
            
            # 🎯 调试：打印每个原始片段的长度和基本信息
            total_speech_duration = 0
            for i, segment in enumerate(speech_segments):
                duration = len(segment) / 16000
                energy = np.mean(np.abs(segment))
                total_speech_duration += duration
                logger.info(f"原始片段 {i+1}: {duration:.2f}秒, {len(segment)}样本, 能量: {energy:.6f}")
            
            speech_ratio = total_speech_duration / audio_duration if audio_duration > 0 else 0
            logger.info(f"语音统计: 总语音时长 {total_speech_duration:.2f}秒, 语音占比 {speech_ratio:.1%}")
            
            # 5. 智能合并短片段以满足 Whisper 要求
            merged_segments = self._merge_short_segments(speech_segments, min_duration=15.0)
            logger.info(f"片段合并后: {len(speech_segments)} → {len(merged_segments)} 个片段")
            
            # 🎯 特殊逻辑：如果合并后片段减少很多，说明有很多短语音
            if len(merged_segments) < len(speech_segments) * 0.5:
                logger.info("🔊 检测到大量短语音片段，可能是重叠说话场景，确保全部转录")
            
            # 如果合并后还是没有有效片段，尝试直接使用原始音频
            if not merged_segments:
                logger.warning("合并后无有效片段，尝试使用原始音频")
                merged_segments = [self._pad_to_min_duration(audio_np, 15.0)]
            
            # 🎯 最终验证：确保所有片段都满足Whisper要求
            validated_segments = []
            for i, segment in enumerate(merged_segments):
                segment_duration = len(segment) / 16000
                if segment_duration < 30.0:  # Whisper要求30秒
                    logger.info(f"最终填充片段 {i+1}: {segment_duration:.2f}秒 → 30.00秒")
                    segment = self._pad_to_min_duration(segment, 30.0)
                validated_segments.append(segment)
                logger.info(f"✅ 最终片段 {i+1}: {len(segment)/16000:.2f}秒")
            
            # 6. 转录处理 - 信任所有VAD检测到的片段
            if len(validated_segments) == 1:
                # 单个片段直接处理
                logger.info("📦 单一片段，直接转录")
                transcription, detected_language = self._transcribe_chunk(
                    validated_segments[0], forced_language, is_first_chunk=True, prompt_text=prompt_text
                )
            else:
                # 多个片段使用上下文传递
                logger.info("📦 多片段，使用上下文转录")
                transcription, detected_language = self._transcribe_vad_segments(
                    validated_segments, forced_language, prompt_text
                )
            
            # 7. 更新上下文缓存（如果启用）
            if use_context and session_id and transcription and transcription.strip():
                self._update_context_cache(session_id, transcription, detected_language)
            
            total_time = time.time() - start_time
            
            # 🎯 性能统计
            if transcription and transcription.strip():
                chars_per_second = len(transcription) / total_time if total_time > 0 else 0
                logger.info(f"✅ 转录完成 - 语言: {detected_language}, "
                        f"音频时长: {audio_duration:.2f}秒, "
                        f"处理耗时: {total_time:.2f}秒, "
                        f"转录速度: {chars_per_second:.1f}字/秒")
            else:
                logger.warning(f"⚠️ 转录完成但无有效结果 - 语言: {detected_language}, "
                            f"音频时长: {audio_duration:.2f}秒, "
                            f"处理耗时: {total_time:.2f}秒")
            
            return transcription, detected_language
            
        except Exception as e:
            logger.error(f"❌ 转录失败: {str(e)}")
            return f"转录失败: {str(e)}", "error"

    def _merge_short_segments(self, segments: list, min_duration: float = 15.0) -> list:
        """智能合并短片段以满足 Whisper 要求"""
        if not segments:
            return []
        
        # 如果只有一个片段，直接检查长度
        if len(segments) == 1:
            single_segment = segments[0]
            if len(single_segment) < min_duration * 16000:
                return [self._pad_to_min_duration(single_segment, min_duration)]
            return segments
        
        merged_segments = []
        current_batch = [segments[0]]
        current_total_duration = len(segments[0]) / 16000
        
        for i in range(1, len(segments)):
            segment = segments[i]
            segment_duration = len(segment) / 16000
            
            # 如果合并后不超过25秒，就继续合并
            if current_total_duration + segment_duration <= 25.0:
                current_batch.append(segment)
                current_total_duration += segment_duration
            else:
                # 合并当前批次
                if current_batch:
                    merged_segment = self._merge_batch_segments(current_batch)
                    merged_segments.append(merged_segment)
                
                # 开始新的批次
                current_batch = [segment]
                current_total_duration = segment_duration
        
        # 处理最后一批
        if current_batch:
            merged_segment = self._merge_batch_segments(current_batch)
            merged_segments.append(merged_segment)
        
        # 确保所有片段都满足最小长度
        final_segments = []
        for segment in merged_segments:
            if len(segment) < min_duration * 16000:
                padded_segment = self._pad_to_min_duration(segment, min_duration)
                final_segments.append(padded_segment)
            else:
                final_segments.append(segment)
        
        logger.info(f"智能合并完成: {len(segments)} → {len(final_segments)} 个片段")
        
        # 调试信息：打印每个最终片段的长度
        for i, segment in enumerate(final_segments):
            duration = len(segment) / 16000
            logger.info(f"最终片段 {i+1}: {duration:.2f}秒")
        
        return final_segments

    def _merge_batch_segments(self, batch_segments: list) -> np.ndarray:
        """合并一批片段，添加适当的静音间隔"""
        if not batch_segments:
            return np.array([])
        
        if len(batch_segments) == 1:
            return batch_segments[0]
        
        merged_segments = []
        for i, segment in enumerate(batch_segments):
            merged_segments.append(segment)
            # 在片段之间添加0.3秒静音（除了最后一个）
            if i < len(batch_segments) - 1:
                silence = np.zeros(int(0.3 * 16000))
                merged_segments.append(silence)
        
        return np.concatenate(merged_segments)

    def _pad_to_min_duration(self, audio_np: np.ndarray, min_duration: float) -> np.ndarray:
        """将音频填充到最小持续时间"""
        target_samples = int(min_duration * 16000)
        
        if len(audio_np) >= target_samples:
            return audio_np
        
        # 计算需要填充的静音长度
        padding_needed = target_samples - len(audio_np)
        
        # 在前后各填充一半的静音
        pad_before = padding_needed // 2
        pad_after = padding_needed - pad_before
        
        padded_audio = np.pad(audio_np, (pad_before, pad_after), mode='constant')
        logger.info(f"音频填充: {len(audio_np)/16000:.2f}秒 → {min_duration}秒")
        
        return padded_audio

    def _transcribe_vad_segments(self, segments: list, forced_language: str = None, 
                               initial_prompt: str = "") -> tuple:
        """转录 VAD 分割的多个语音片段"""
        all_transcriptions = []
        detected_language = "auto"
        current_prompt = initial_prompt
        
        for i, segment in enumerate(segments):
            segment_duration = len(segment) / 16000
            logger.info(f"处理语音片段 {i+1}/{len(segments)} (时长: {segment_duration:.2f}秒)")
            
            if current_prompt:
                logger.info(f"📝 使用提示词: {current_prompt[-100:]}...")
            
            segment_transcription, segment_language = self._transcribe_chunk(
                segment, forced_language, 
                is_first_chunk=(i == 0),
                prompt_text=current_prompt
            )
            
            if segment_transcription and segment_transcription.strip():
                all_transcriptions.append(segment_transcription)
                
                # 更新提示词：将当前识别结果作为下一段的提示
                current_prompt = self._truncate_prompt(current_prompt + " " + segment_transcription)
                logger.info(f"✅ 片段 {i+1} 转录成功，更新提示词")
            else:
                logger.warning(f"⚠️ 片段 {i+1} 无有效转录")
                # 即使转录失败，也保留之前的提示词
            
            # 记录第一个片段检测到的语言
            if i == 0 and segment_language and segment_language != "auto":
                detected_language = segment_language
        
        # 合并结果
        final_transcription = self._merge_transcriptions(all_transcriptions)
        logger.info(f"🔗 合并完成: {len(all_transcriptions)}个片段 -> {len(final_transcription)}字符")
        
        return final_transcription, detected_language

    def _transcribe_chunk(self, chunk: np.ndarray, forced_language: str = None, 
                         is_first_chunk: bool = False, prompt_text: str = "") -> tuple:
        """转录单个音频块 - 完整修复版"""
        try:
            # 🎯 关键修复：在开始时严格验证输入长度
            chunk_duration = len(chunk) / 16000
            logger.info(f"转录开始验证: 输入音频{chunk_duration:.2f}秒, {len(chunk)}样本")
            
            # 根据实际测试，Whisper-large-v3 需要30秒输入
            REQUIRED_DURATION = 30.0
            if chunk_duration < REQUIRED_DURATION:
                logger.warning(f"输入音频不足30秒: {chunk_duration:.2f}秒, 强制填充到30秒")
                chunk = self._pad_to_min_duration(chunk, REQUIRED_DURATION)
                chunk_duration = len(chunk) / 16000
                logger.info(f"填充后: {chunk_duration:.2f}秒")
            
            # 🎯 关键修改：极宽松的能量检测，只过滤完全静音
            energy = np.mean(np.abs(chunk))
            logger.info(f"音频能量参考值: {energy:.6f}")
            
            # 只过滤能量接近0的完全静音
            if energy < 0.0001:
                logger.warning(f"能量极低，可能为误检静音 (能量: {energy:.6f})")
                return "", "auto"
            
            # 🎯 特殊提示：中等能量可能是重叠说话
            if energy < 0.005:
                logger.info(f"检测到可能的重叠说话或轻语音 (能量: {energy:.6f})，继续转录")

            generation_params = {
                "task": "transcribe",
                "num_beams": 1,
                "temperature": 0.0,
                "no_repeat_ngram_size": 3,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            if forced_language and forced_language != "auto":
                generation_params["language"] = forced_language
            
            # 兼容的提示词处理方式
            if prompt_text and prompt_text.strip():
                try:
                    # 方式1：使用 text 参数（新版本 transformers）
                    inputs = self.processor(
                        chunk,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                        text=prompt_text
                    )
                    logger.debug(f"使用提示词(新方式): {prompt_text[:50]}...")
                except Exception as e:
                    logger.warning(f"新提示词方式失败: {e}，尝试传统方式")
                    # 方式2：传统方式
                    inputs = self.processor(
                        chunk,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )
                    # 在生成参数中设置提示词
                    prompt_tokens = self.processor(text=prompt_text, return_tensors="pt").input_ids
                    generation_params["forced_decoder_ids"] = self._create_forced_decoder_ids(prompt_tokens)
            else:
                inputs = self.processor(
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
            
            input_features = inputs["input_features"].to(self.device, dtype=torch.float16)
            
            with torch.inference_mode():
                generated_ids = self.whisper_model.generate(input_features, **generation_params)
            
            raw_transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 过滤空结果
            if not raw_transcription.strip() or len(raw_transcription.strip()) < 2:
                logger.warning("转录结果为空或过短")
                return "", "auto"
            
            detected_language = "auto"
            if is_first_chunk and not forced_language:
                detected_language = self._detect_language_from_ids(generated_ids)
            
            # 使用后处理器清理和矫正文本
            transcription = self.post_processor.clean_text(raw_transcription)
            
            # 清理内存
            del input_features, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"片段转录结果: {transcription[:100]}...")
            return transcription, detected_language
            
        except Exception as e:
            logger.error(f"音频片段转录失败: {e}")
            # 尝试不使用提示词重新转录
            if prompt_text:
                logger.info("尝试不使用提示词重新转录...")
                try:
                    return self._transcribe_chunk(chunk, forced_language, is_first_chunk, "")
                except:
                    return "", "auto"
            return "", "auto"

    def _create_forced_decoder_ids(self, prompt_tokens):
        """创建强制解码器ID（兼容旧版本）"""
        forced_decoder_ids = []
        for i, token_id in enumerate(prompt_tokens[0]):
            forced_decoder_ids.append([i + 1, token_id])
        return forced_decoder_ids

    def _truncate_prompt(self, prompt: str, max_tokens: int = 200) -> str:
        """截断提示词以避免过长"""
        if not prompt or len(prompt) < 50:
            return prompt
        
        max_chars = max_tokens * 3
        if len(prompt) <= max_chars:
            return prompt
        
        truncated = prompt[-max_chars:]
        sentence_breaks = ['。', '！', '？', '.', '!', '?', '，', ',']
        
        for break_char in sentence_breaks:
            break_pos = truncated.find(break_char)
            if break_pos > 10:
                truncated = truncated[break_pos + 1:].lstrip()
                break
        
        logger.debug(f"提示词截断: {len(prompt)} -> {len(truncated)} 字符")
        return truncated

    def _merge_transcriptions(self, transcriptions: list) -> str:
        """合并多个转录结果"""
        if not transcriptions:
            return "未检测到语音"
        
        merged = " ".join(transcriptions)
        
        # 基础去重
        words = merged.split()
        if len(words) > 10:
            for i in range(len(words) - 6):
                if words[i:i+3] == words[i+3:i+6]:
                    merged = " ".join(words[:i+3] + words[i+6:])
                    logger.info("检测并移除了重复内容")
                    break
        
        return merged

    def _update_context_cache(self, session_id: str, transcription: str, language: str):
        """更新上下文缓存"""
        max_sessions = 100
        max_length = 800
        
        if len(self.context_cache) >= max_sessions:
            first_key = next(iter(self.context_cache))
            del self.context_cache[first_key]
            logger.info(f"上下文缓存已满，移除会话: {first_key}")
        
        truncated_text = transcription[:max_length] if len(transcription) > max_length else transcription
        self.context_cache[session_id] = truncated_text
        logger.info(f"更新会话 {session_id} 的上下文缓存: {len(truncated_text)} 字符")

    def clear_context_cache(self, session_id: str = None):
        """清理上下文缓存"""
        if session_id:
            if session_id in self.context_cache:
                del self.context_cache[session_id]
                logger.info(f"已清理会话 {session_id} 的上下文缓存")
        else:
            self.context_cache.clear()
            logger.info("已清理所有上下文缓存")

    def _detect_language_from_ids(self, generated_ids: torch.Tensor) -> str:
        """从生成的token中检测语言"""
        try:
            if generated_ids.numel() == 0:
                return "auto"
                
            first_tokens = generated_ids[0][:2]
            decoded = self.processor.decode(first_tokens)
            
            if '<|en|>' in decoded:
                return "en"
            elif '<|zh|>' in decoded:
                return "zh" 
            elif '<|yue|>' in decoded:
                return "yue"
            else:
                return "auto"
                
        except Exception as e:
            logger.warning(f"语言检测失败: {e}")
            return "auto"

    def clear_gpu_memory(self):
        """清理GPU内存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("✅ GPU内存已清理")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        if not self.model_loaded:
            return {"status": "模型未加载"}
        
        info = {
            "status": "模型已加载",
            "model_name": "Whisper-large-v3",
            "device": str(self.device),
            "sample_rate": "16kHz",
            "supported_languages": ["auto", "yue", "zh", "en"],
            "processing_flow": "VAD语音分割 → 片段合并 → Whisper转录 → 词汇矫正",
            "vocabulary_stats": self.get_vocabulary_stats(),
            "context_cache_size": len(self.context_cache),
            "features": [
                "Silero VAD语音检测",
                "智能片段合并", 
                "提示词上下文连贯性", 
                "自动语言检测",
                "统一词汇矫正"
            ]
        }
        
        if self.device.type == "cuda":
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.1f}G"
            
        return info

    # 词汇处理相关方法保持不变
    def add_custom_correction(self, wrong: str, correct: str):
        """添加自定义词汇矫正"""
        self.post_processor.add_custom_correction(wrong, correct)

    def batch_add_corrections(self, corrections: dict):
        """批量添加词汇矫正"""
        self.post_processor.batch_add_corrections(corrections)

    def remove_correction(self, word: str):
        """移除词汇矫正"""
        self.post_processor.remove_correction(word)

    def get_vocabulary_stats(self):
        """获取词汇表统计"""
        return self.post_processor.get_vocabulary_stats()

    def search_corrections(self, keyword: str):
        """搜索相关矫正项"""
        return self.post_processor.search_corrections(keyword)