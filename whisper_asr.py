import torch
import logging
import time
import traceback
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from silero_vad import load_silero_vad, get_speech_timestamps

from config import MODEL_PATH, OPTIMIZATION_CONFIG, DEVICE
from audio_processor import AudioProcessor
from post_processor import PostProcessor

logger = logging.getLogger(__name__)

class WhisperASR:
    """Whisper ASR 核心类 - 完全自动语言识别"""
    
    def __init__(self):
        """初始化 WhisperASR"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")

        self.device = DEVICE
        torch.cuda.set_device(0)
        logger.info(f"使用设备: {self.device}, GPU: {torch.cuda.get_device_name(0)}")

        self.whisper_model = None
        self.processor = None
        self.vad_model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
        # 初始化处理模块
        self.audio_processor = AudioProcessor(sample_rate=OPTIMIZATION_CONFIG["sample_rate"])
        self.post_processor = PostProcessor()
        
        # 配置
        self.optimization_config = OPTIMIZATION_CONFIG
        self.SAMPLE_RATE = OPTIMIZATION_CONFIG["sample_rate"]
        
        self.supported_languages = [
            'English', 'Chinese', 'Cantonese', 'Japanese', 'Korean', 'Vietnamese'
        ]

    def log_memory(self, stage: str):
        """记录 GPU 内存使用情况"""
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            free_memory = total_memory - allocated_memory
            logger.info(f"{stage}: 可用 {free_memory:.2f} GiB / 总共 {total_memory:.2f} GiB")

    def clear_gpu_memory(self):
        """清理 GPU 内存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.log_memory("清理后内存")

    def load_models(self, model_path=MODEL_PATH):
        """加载 Whisper 和 VAD 模型"""
        try:
            self.clear_gpu_memory()
            logger.info("🔄 加载 Whisper 模型...")
            self.processor = WhisperProcessor.from_pretrained(
                model_path, 
                num_mel_bins=self.optimization_config["n_mels"]
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
            self.whisper_model.eval()
            self.log_memory("Whisper 模型加载后内存")
            
            logger.info("🔄 加载 Silero VAD 模型...")
            self.vad_model = load_silero_vad(onnx=True)
            self.vad_get_speech_timestamps = get_speech_timestamps
            logger.info("✅ Silero VAD 模型加载完成")
            
            self.model_loaded = True
            self.log_memory("所有模型加载后内存")
            logger.info("✅ 所有模型加载完成")
            self.warmup_model()
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            self.model_loaded = False
            raise

    def warmup_model(self):
        """模型预热，提高首次推理速度"""
        logger.info("🔥 预热模型...")
        try:
            t = np.linspace(0, 1, self.SAMPLE_RATE)
            dummy_audio = 0.1 * np.sin(2 * np.pi * 300 * t)
            dummy_audio = dummy_audio.astype(np.float32)
            inputs = self.processor(
                dummy_audio,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(self.device)
            model_dtype = next(self.whisper_model.parameters()).dtype
            input_features = input_features.to(model_dtype)
            with torch.no_grad():
                # 预热多种语言
                _ = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                    max_length=10,
                    num_beams=1
                )
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}, 继续启动")

    def detect_language(self, audio_np, segment_duration=5.0, retries=3):
        """完全自动语言检测"""
        try:
            sample_length = min(int(segment_duration * self.SAMPLE_RATE), len(audio_np))
            if sample_length < 8000:  # 确保至少0.5秒音频
                sample_length = min(8000, len(audio_np))
            sample_audio = audio_np[:sample_length]
            
            inputs = self.processor(
                sample_audio,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(self.device)
            model_dtype = next(self.whisper_model.parameters()).dtype
            input_features = input_features.to(model_dtype)
            
            with torch.no_grad():
                # 让模型自动检测语言
                generated_ids = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                    max_new_tokens=10,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # 从生成的token中提取语言信息
                lang_tokens = generated_ids.sequences[0]
                detected_languages = []
                
                for token in lang_tokens:
                    lang_code = self.processor.tokenizer.convert_ids_to_tokens([token])[0]
                    lang_code = re.sub(r'[<>]', '', lang_code).strip().lower()
                    if lang_code in self.supported_languages:
                        detected_languages.append(lang_code)
                
                logger.info(f"自动语言检测结果: {detected_languages}")
                
                # 返回检测到的第一个有效语言
                if detected_languages:
                    detected_lang = detected_languages[0]
                    logger.info(f"自动检测到语言: {detected_lang}")
                    return detected_lang
                
                if retries > 0:
                    logger.warning(f"语言检测结果为空，剩余重试次数: {retries}，尝试更短片段")
                    return self.detect_language(audio_np, segment_duration=segment_duration/2, retries=retries-1)
                
                logger.warning("未检测到有效语言，使用自动模式")
                return "auto"
                
        except Exception as e:
            logger.warning(f"语言检测失败: {e}，剩余重试次数: {retries}")
            if retries > 0:
                return self.detect_language(audio_np, segment_duration=segment_duration/2, retries=retries-1)
            logger.warning("语言检测失败，使用自动模式")
            return "auto"

    def optimize_vad_parameters(self, audio_tensor):
        """优化VAD参数"""
        energy = torch.mean(torch.abs(audio_tensor))
        energy_std = torch.std(torch.abs(audio_tensor))
        logger.info(f"音频能量水平: {energy:.4f}, 能量标准差: {energy_std:.4f}")
        
        if energy < 0.002:
            return {
                "threshold": 0.05,
                "min_silence_duration_ms": 50,
                "min_speech_duration_ms": 50,
                "speech_pad_ms": 10
            }
        elif energy < 0.015:
            return {
                "threshold": 0.02,
                "min_silence_duration_ms": 150,
                "min_speech_duration_ms": 80,
                "speech_pad_ms": 20
            }
        elif energy > 0.08:
            return {
                "threshold": 0.20,
                "min_silence_duration_ms": 100,
                "min_speech_duration_ms": 150,
                "speech_pad_ms": 50
            }
        else:
            return {
                "threshold": 0.05,
                "min_silence_duration_ms": 60,
                "min_speech_duration_ms": 100,
                "speech_pad_ms": 30
            }

    def adaptive_vad_processing(self, audio_tensor):
        """自适应VAD处理"""
        vad_params = self.optimize_vad_parameters(audio_tensor)
        logger.info(f"使用VAD参数: {vad_params}")
        
        speech_timestamps = self.vad_get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=self.SAMPLE_RATE,
            **vad_params,
            return_seconds=False
        )
        
        if not speech_timestamps:
            logger.info("尝试更宽松的VAD参数...")
            fallback_params = [
                {"threshold": 0.008, "min_silence_duration_ms": 30, "min_speech_duration_ms": 50, "speech_pad_ms": 10},
                {"threshold": 0.01, "min_silence_duration_ms": 20, "min_speech_duration_ms": 30, "speech_pad_ms": 5},
                {"threshold": 0.005, "min_silence_duration_ms": 10, "min_speech_duration_ms": 20, "speech_pad_ms": 5},
                {"threshold": 0.001, "min_silence_duration_ms": 10, "min_speech_duration_ms": 10, "speech_pad_ms": 5}
            ]
            
            for params in fallback_params:
                logger.info(f"使用回退VAD参数: {params}")
                speech_timestamps = self.vad_get_speech_timestamps(
                    audio_tensor,
                    self.vad_model,
                    sampling_rate=self.SAMPLE_RATE,
                    **params,
                    return_seconds=False
                )
                if speech_timestamps:
                    break
        
        # 强制分段
        max_segment_duration = 30 * self.SAMPLE_RATE  # 30秒
        final_timestamps = []
        for segment in speech_timestamps:
            start = segment['start']
            end = segment['end']
            segment_duration = end - start
            if segment_duration > max_segment_duration:
                num_splits = int(np.ceil(segment_duration / max_segment_duration))
                split_duration = segment_duration // num_splits
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(split_start + split_duration, end)
                    final_timestamps.append({'start': split_start, 'end': split_end})
            else:
                final_timestamps.append(segment)
        
        logger.info(f"VAD 检测到 {len(final_timestamps)} 个语音片段")
        return final_timestamps

    def optimize_generation_parameters(self, audio_duration, audio_quality=0.5, detected_language=None):
        """优化生成参数 - 基于检测到的语言自动优化"""
        base_params = {
            "max_length": int(audio_duration * 60),
            "num_beams": 12,
            "temperature": 0.2,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "repetition_penalty": 1.2
        }
        
        # 基于检测到的语言进行优化
        if detected_language == "yue":
            base_params.update({
                "temperature": 0.15,
                "num_beams": 12,
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 4
            })
        elif detected_language == "en":
            base_params.update({
                "temperature": 0.15,
                "num_beams": 12,
                "repetition_penalty": 1.25,
                "no_repeat_ngram_size": 3
            })
        elif detected_language == "zh":
            base_params.update({
                "temperature": 0.2,
                "num_beams": 10,
                "repetition_penalty": 1.2
            })
        
        # 基于音频时长调整
        if audio_duration > 60:
            base_params.update({
                "max_length": int(audio_duration * 80),
                "num_beams": 8
            })
        elif audio_duration < 5:
            base_params.update({
                "max_length": 256,
                "num_beams": 6,
                "early_stopping": True
            })
        
        # 基于音频质量调整
        if audio_quality < 0.4:
            base_params.update({
                "num_beams": 14,
                "repetition_penalty": 1.5
            })
            
        return base_params

    def transcribe_segment(self, segment, segment_duration, detected_language="auto", audio_quality=0.5):
        """转录单个音频片段"""
        try:
            gen_params = self.optimize_generation_parameters(segment_duration, audio_quality, detected_language)
            
            inputs = self.processor(
                segment,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(self.device)
            model_dtype = next(self.whisper_model.parameters()).dtype
            input_features = input_features.to(model_dtype)
            
            with torch.no_grad():
                # 让模型自动处理语言
                if detected_language != "auto":
                    gen_params["language"] = detected_language
                
                generated_ids = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                    **gen_params
                )
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 根据检测到的语言进行后处理
            final_transcription = self.clean_text(transcription, detected_language)
            return final_transcription
            
        except Exception as e:
            logger.error(f"片段转录失败: {e}")
            return ""

    def transcribe_with_vad(self, audio_np, audio_duration, audio_quality=0.5):
        """使用VAD分段转录"""
        logger.info("使用VAD分段处理音频")
        audio_tensor = torch.from_numpy(audio_np).float().to("cpu")
        if len(audio_tensor.shape) > 1:
            audio_tensor = torch.mean(audio_tensor, dim=1)
            
        speech_timestamps = self.adaptive_vad_processing(audio_tensor)
        all_transcriptions = []
        
        for i, segment_info in enumerate(speech_timestamps):
            start_sample = segment_info['start']
            end_sample = segment_info['end']
            segment = audio_np[start_sample:end_sample]
            segment_duration = (end_sample - start_sample) / self.SAMPLE_RATE
            
            logger.info(f"处理片段 {i+1}/{len(speech_timestamps)}: 时长: {segment_duration:.2f}秒")
            
            # 为每个片段检测语言
            segment_language = self.detect_language(segment, segment_duration=min(1.0, segment_duration))
            transcription = self.transcribe_segment(segment, segment_duration, segment_language, audio_quality)
            
            if transcription:
                all_transcriptions.append(transcription)
                logger.info(f"片段 {i+1} 转录结果: {transcription} (语言: {segment_language})")
            
            torch.cuda.empty_cache()
        
        final_transcription = " ".join(all_transcriptions)
        return final_transcription, "mixed"

    def transcribe_full_audio(self, audio_np, audio_duration, audio_quality=0.5):
        """直接处理完整音频"""
        logger.info("🔄 直接处理完整音频...")
        
        # 检测整体语言
        detected_language = self.detect_language(audio_np)
        logger.info(f"完整音频检测到语言: {detected_language}")
        
        # 分块处理长音频
        chunk_duration = 30  # 每块30秒
        num_chunks = int(np.ceil(audio_duration / chunk_duration))
        all_transcriptions = []
        
        for i in range(num_chunks):
            start_sample = int(i * chunk_duration * self.SAMPLE_RATE)
            end_sample = min(int((i + 1) * chunk_duration * self.SAMPLE_RATE), len(audio_np))
            chunk = audio_np[start_sample:end_sample]
            chunk_duration_actual = (end_sample - start_sample) / self.SAMPLE_RATE
            
            logger.info(f"处理音频块 {i+1}/{num_chunks}: 时长: {chunk_duration_actual:.2f}秒")
            
            transcription = self.transcribe_segment(chunk, chunk_duration_actual, detected_language, audio_quality)
            
            if transcription:
                all_transcriptions.append(transcription)
            
            torch.cuda.empty_cache()
        
        final_transcription = " ".join(all_transcriptions)
        logger.info(f"完整音频转录结果: {final_transcription}")
        return final_transcription, detected_language

    def transcribe_audio(self, audio_bytes, forced_language=None):
        """转录完整音频 - 完全自动语言识别"""
        if not self.model_loaded:
            logger.error("模型未加载")
            return "模型未加载，请检查服务状态", "unknown"
        
        try:
            start_time = time.time()
            
            # 处理音频
            audio_np = self.process_audio(audio_bytes)
            audio_np = self.enhance_telephone_audio(audio_np)
            audio_duration = len(audio_np) / self.SAMPLE_RATE
            audio_quality = self.estimate_audio_quality(audio_np)
            
            logger.info(f"增强后音频时长: {audio_duration:.2f}秒, 质量: {audio_quality:.2f}")
            
            # 完全自动语言处理
            if audio_duration > 10:  # 长音频使用VAD分段
                transcription, final_language = self.transcribe_with_vad(audio_np, audio_duration, audio_quality)
            else:  # 短音频直接处理
                transcription, final_language = self.transcribe_full_audio(audio_np, audio_duration, audio_quality)
            
            logger.info(f"转录总耗时: {time.time() - start_time:.2f}秒")
            return transcription if transcription else "未检测到语音", final_language
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            traceback.print_exc()
            return f"转录失败: {str(e)}", "error"

    # 使用新的音频处理器和后处理器的方法
    def process_audio(self, audio_bytes):
        """处理音频数据 - 使用新的音频处理器"""
        return self.audio_processor.process_audio(audio_bytes)

    def estimate_audio_quality(self, audio_np):
        """估计音频质量 - 使用新的音频处理器"""
        return self.audio_processor.estimate_audio_quality(audio_np)

    def enhance_telephone_audio(self, audio_np):
        """增强电话录音质量 - 使用新的音频处理器"""
        return self.audio_processor.enhance_telephone_audio(audio_np)

    def clean_text(self, text, language=None):
        """清理转录文本 - 使用新的后处理器"""
        return self.post_processor.clean_text(text, language)

    def add_custom_correction(self, language: str, wrong: str, correct: str):
        """添加自定义修正"""
        self.post_processor.add_custom_correction(language, wrong, correct)

    def batch_add_corrections(self, language: str, corrections: dict):
        """批量添加修正"""
        self.post_processor.batch_add_corrections(language, corrections)