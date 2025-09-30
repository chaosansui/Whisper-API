import torch
import numpy as np
import traceback
import re
import logging
import os
import time
import torchaudio
import math # 导入 math 库用于 RMS 计算
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from preprocessing.audio_enhancer import AudioEnhancer
from preprocessing.vad_segmenter import VADSegmenter, VAD_SAMPLE_RATE
from preprocessing.diarization_core import DiarizationCore 
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple
from scipy.signal import butter, filtfilt
import io
from config import MODEL_PATH, SAMPLE_RATE, SPEAKER_EMBEDDING_PATH

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = -0.7 

RMS_ENERGY_THRESHOLD = 0.015 

def process_audio(audio_bytes: bytes) -> np.ndarray:
    """将音频字节流转换为单声道 16kHz NumPy 数组"""
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.shape[0] > 1:
            # 转换为单声道
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            # 重采样至目标采样率
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
        audio_np = waveform.squeeze().numpy()
        if np.max(np.abs(audio_np)) < 1e-3:
            logger.warning("音频幅度过低，可能为静音")
        return audio_np
    except Exception as e:
        logger.error(f"音频处理失败: {e}")
        raise

def calculate_rms_energy(audio_segment: np.ndarray) -> float:
    """计算音频片段的 RMS (Root Mean Square) 能量。"""
    if len(audio_segment) == 0:
        return 0.0
    # RMS = sqrt(mean(x^2))
    return np.sqrt(np.mean(audio_segment**2))


class WhisperASR:
    def __init__(self, hf_token: Optional[str] = None):
        if not torch.cuda.is_available():
            logger.error("CUDA 不可用，尝试使用 CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0") 
            torch.cuda.set_device(0)
            logger.info(f"使用设备: {self.device}, GPU: {torch.cuda.get_device_name(0)}")
        
        self.torch = torch
        self.whisper_model = None
        self.processor = None
        self.enhancer = None
        self.vad_segmenter = None
        self.diarization_core = None
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        self.SAMPLE_RATE = SAMPLE_RATE
        self.hf_token = hf_token

        self.optimization_config = {
            "use_flash_attention": False,
            "chunk_length": 30,
            "batch_size": 1,
            "max_retries": 5,
            "timeout": 60.0,
            "n_mels": 128
        }
        
        # 移除 language_mapping，仅保留支持的语言列表
        self.supported_languages = ['en', 'zh', 'yue', 'auto'] 

    def log_memory(self, stage: str):
        """记录GPU内存使用情况"""
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            free_memory = total_memory - allocated_memory
            logger.debug(f"{stage}: 可用 {free_memory:.2f} GiB / 总共 {total_memory:.2f} GiB")

    def clear_gpu_memory(self):
        """清理GPU内存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.log_memory("清理后内存")

    def load_models(self):
        """加载所有模型"""
        try:
            self.clear_gpu_memory()
            logger.debug(f"🔄 尝试加载 Whisper 模型: {MODEL_PATH}")
            
            # 加载 Whisper 模型
            self.processor = WhisperProcessor.from_pretrained(MODEL_PATH)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
            self.whisper_model.eval()
            logger.info("✅ Whisper 模型加载成功")

            # 【加载模块化前处理模型】
            logger.debug("🔄 尝试加载模块化的 Diarization 依赖...")
            
            # 1. 初始化 Enhancer (流程 1, 2, 4)
            self.enhancer = AudioEnhancer(sample_rate=self.SAMPLE_RATE)
            
            # 2. 初始化 VAD Segmenter (流程 3) - VAD 在 CPU 上运行，使用 16kHz
            self.vad_segmenter = VADSegmenter(sample_rate=VAD_SAMPLE_RATE, device='cpu')
            
            # 3. 初始化 Diarization Core (流程 5, 8, 9) - 运行在 GPU 上
            self.diarization_core = DiarizationCore(
                speaker_embedding_path=SPEAKER_EMBEDDING_PATH,
                sample_rate=self.SAMPLE_RATE,
                device=self.device
            )

            logger.info("✅ 模块化 Diarization 依赖加载完成")
            self.model_loaded = True
            logger.info("✅ 所有模型加载完成")
            self.warmup_model()

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            logger.error(f"详细堆栈: {traceback.format_exc()}")
            self.model_loaded = False
            raise

    def warmup_model(self):
        """预热模型"""
        logger.debug("🔥 预热模型...")
        try:
            t = np.linspace(0, 1, self.SAMPLE_RATE).astype(np.float32)
            dummy_audio = 0.1 * np.sin(2 * np.pi * 300 * t)
            inputs = self.processor(
                dummy_audio,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_features = inputs["input_features"].to(self.device, dtype=torch.float16)
            attention_mask = inputs.get("attention_mask").to(self.device) if inputs.get("attention_mask") is not None else None
            
            with torch.no_grad():
                for lang in ["yue", "en"]:
                    _ = self.whisper_model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        task="transcribe",
                        max_length=10,
                        num_beams=1,
                        language=lang
                    )
            
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}, 继续启动")

    def detect_language(self, audio_np: np.ndarray, segment_duration: float = 5.0) -> str:
        """检测音频语言，适配 whisper-large-v3。
        由于模型的语言检测能力，此方法暂被 stub 替代。"""
        logger.warning("语言检测暂时跳过，回退到粤语 (yue) 或使用 'auto'")
        return "yue" 

    def map_speaker_to_role(self, text: str) -> str:
        """根据文本内容判断说话人角色"""
        agent_keywords = ["你好", "欢迎致电", "客服", "有什么可以帮助", "hello", "hi", "welcome"]
        return "客服" if any(keyword in text.lower() for keyword in agent_keywords) else "客户"

    def _post_process_text(self, text: str, language: str) -> str:
        """
        流程 10: 清理转录文本，去除停顿词、重复标点、修正句末标点等。
        """
        if not text:
            return ""

        # 1. 移除常见的杂音标记和停顿词
        text = re.sub(r'\[\s*(laughter|music|sigh|whispering|cough)\s*\]', '', text, flags=re.IGNORECASE)
        # 移除常见的“呃”或“嗯”等非语义停顿
        text = re.sub(r'\s*呃(呃)*\s*|\s*嗯(嗯)*\s*', ' ', text) 

        # 2. 移除重复的标点符号和多余空格
        # 合并多个相同的标点符号
        text = re.sub(r'([，。！？])\s*\1+', r'\1', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3. 修正 Whisper 有时在开头加入的重复短语（例如 "你好 你好" -> "你好"）
        words = text.split()
        if len(words) > 3 and words[0] == words[1] and words[0] != words[2]:
            text = ' '.join(words[1:])

        # 4. 确保中文或粤语句尾有标点
        if language in ['zh', 'yue'] and text and text[-1] not in ['。', '！', '？', '...']:
            text += '。'
            
        return text

    def transcribe_segment(self, audio_np: np.ndarray, duration: float, language: str) -> Tuple[str, float]:
        """
        [流程 6] 转录单个音频片段。
        返回值: (transcription: str, confidence_score: float)
        """
        try:
            # === 新增：RMS 能量过滤替代置信度过滤 ===
            rms_energy = calculate_rms_energy(audio_np)
            if rms_energy < RMS_ENERGY_THRESHOLD:
                logger.warning(f"音频能量过滤：RMS {rms_energy:.4f} 低于阈值 {RMS_ENERGY_THRESHOLD}，跳过转录。")
                # 返回空文本和高置信度 (1.0)，确保在后续流程中它被视为有效过滤（即，无文本）
                return "", 1.0 
            
            # 移除旧的、不精确的幅度检查
            # if np.max(np.abs(audio_np)) < 1e-3:
            #     logger.warning("音频幅度过低，跳过转录")
            #     return "", 1.0 
            
            # 使用 'auto' 语言选项，让 Whisper 自行检测（除非强制指定）
            if language == 'auto':
                lang_param = None
            else:
                lang_param = language

            gen_params = {
                "max_length": min(int(duration * 30), 200),
                "num_beams": 5,
                "temperature": 0.7,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "language": lang_param, # 使用参数
                "task": "transcribe"
                # WARNING FIX: 移除 output_scores=True 和 return_dict_in_generate=True
            }
            
            inputs = self.processor(
                audio_np,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_features = inputs["input_features"].to(self.device, dtype=torch.float16)
            attention_mask = inputs.get("attention_mask").to(self.device) if inputs.get("attention_mask") is not None else None
            
            with torch.no_grad():
                # 警告修复：不再传入 output_scores 和 return_dict_in_generate
                generated_ids = self.whisper_model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **gen_params
                )
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # WARNING FIX: 无法可靠获取置信度，临时设置为 1.0
            confidence_score = 1.0 
            logger.debug(f"ℹ️ 流程 7 (置信度过滤) 已被 RMS 能量过滤替代。转录置信度分数被设置为 {confidence_score}。")
            
            return transcription, confidence_score
        
        except Exception as e:
            logger.warning(f"片段转录失败: {e}")
            return "", -10.0 # 失败时返回极低的置信度，但此路径应该很少触发

    def process_call_recording(self, audio_bytes: bytes, forced_language: Optional[str] = None) -> Tuple[List[Dict], str]:
        """处理通话录音的主方法"""
        try:
            start_time = time.time()
            
            # 1. 初始音频处理
            audio_np = process_audio(audio_bytes)
            audio_duration = len(audio_np) / self.SAMPLE_RATE
            logger.info(f"音频时长: {audio_duration:.2f}秒")

            # 2. 语言检测 (移除 language_mapping 逻辑)
            if forced_language and forced_language in self.supported_languages:
                detected_language = forced_language
            else:
                # 如果未强制指定或指定语言不支持，则使用检测结果 (目前 stub 为 'yue')
                detected_language = self.detect_language(audio_np)
            
            # 确保最终使用一个合理的语言代码，否则回退到 'auto'
            if detected_language not in self.supported_languages:
                 detected_language = "auto"
                 logger.warning(f"无效或未检测到语言，将使用 Whisper 模型的 'auto' 语言检测。")

            # 3. 运行模块化 Diarization 和 Transcribe
            results = self._run_diarization_and_transcription(audio_np, audio_duration, detected_language)
            
            logger.info(f"处理完成，耗时: {time.time() - start_time:.2f}秒")
            return results, detected_language
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            logger.error(f"详细堆栈: {traceback.format_exc()}")
            return [{
                "role": "错误",
                "speaker": "None",
                "text": f"处理失败: {str(e)}",
                "start": 0.0,
                "end": 0.0
            }], "error"

    def _run_diarization_and_transcription(self, audio_np: np.ndarray, audio_duration: float, language: str) -> List[Dict]:
        """
        [流程 1-5, 8, 9] Diarization & [流程 6] Transcribe
        """
        results = []
        
        if not self.diarization_core or not self.vad_segmenter or not self.enhancer:
            logger.warning("⚠️ Diarization 依赖未加载，使用完整转录")
            return self._transcribe_full_audio(audio_np, audio_duration, language)
        
        if audio_duration < 2.0:
            logger.info("⚠️ 音频过短，使用完整转录")
            return self._transcribe_full_audio(audio_np, audio_duration, language)
        
        try:
            logger.info("🎯 开始模块化前处理和转录...")
            
            # 1. 质量检测 / 降噪 / 增强 (AudioEnhancer)
            self.enhancer.quality_check(audio_np) # 流程 1
            enhanced_audio_np = self.enhancer.denoise_and_enhance(audio_np) # 流程 2 & 4
            
            # 2. VAD 分割 (VADSegmenter)
            # 关键修复：确保 VAD 输入音频是 16kHz
            if self.SAMPLE_RATE != VAD_SAMPLE_RATE:
                logger.info(f"🔄 正在将音频从 {self.SAMPLE_RATE} Hz 重采样至 VAD 所需的 {VAD_SAMPLE_RATE} Hz...")
                
                # 转换 NumPy 数组为 Tensor
                audio_tensor = torch.from_numpy(enhanced_audio_np).float().unsqueeze(0)
                
                # 执行重采样
                resampler = torchaudio.transforms.Resample(self.SAMPLE_RATE, VAD_SAMPLE_RATE)
                # 使用 to("cpu") 确保在 CPU 上进行重采样，减少 GPU 负担
                vad_input_tensor = resampler(audio_tensor.to("cpu")) 
                
                # 转换回 NumPy 数组
                vad_input_np = vad_input_tensor.squeeze().numpy()
            else:
                vad_input_np = enhanced_audio_np
            
            # 流程 3
            # VADSegmenter 现在接收 16kHz 音频，并且具有更短的静音阈值 (在 vad_segmenter.py 中已设置)
            self.vad_segmenter.sample_rate = VAD_SAMPLE_RATE # 临时修正 VADSegmenter 内部的采样率检查
            vad_segments = self.vad_segmenter.get_speech_segments(vad_input_np) 
            logger.info(f"🎯 VAD Segmenter 得到 {len(vad_segments)} 个片段")

            # 3. 说话人核心分离 (DiarizationCore)
            diarized_segments = self.diarization_core.diarize(enhanced_audio_np, vad_segments) # 流程 5, 8, 9
            logger.info(f"🎯 Diarization Core 得到 {len(diarized_segments)} 个带标签的片段")

            if not diarized_segments:
                logger.warning("❌ Diarization 核心未检测到有效片段，回退到完整转录")
                return self._transcribe_full_audio(audio_np, audio_duration, language)

            # 4. 转录每个片段 (流程 6)
            futures = []
            for segment in diarized_segments:
                # 提交任务：_transcribe_single_segment 返回 (transcription, confidence)
                futures.append(self.executor.submit(self._transcribe_single_segment, enhanced_audio_np, segment, language))
            
            successful_transcriptions = 0
            
            for i, future in enumerate(futures):
                transcription, confidence_score = future.result()
                
                # 【流程 7: 分段质量过滤 - 替代方案：检查转录文本是否为空】
                # 由于 RMS 过滤现在在 transcribe_segment 内部执行，如果转录结果为空，
                # 意味着该片段已被 RMS 过滤或模型转录失败。
                if transcription and len(transcription.strip()) > 0:
                    
                    # 【流程 10: 文本后处理】
                    processed_text = self._post_process_text(transcription, language)
                    
                    role = self.map_speaker_to_role(processed_text)
                    
                    # 5. 结果聚合
                    results.append({
                        "role": role,
                        "speaker": diarized_segments[i]["speaker"], 
                        "text": processed_text,
                        "start": diarized_segments[i]["start"],
                        "end": diarized_segments[i]["end"]
                    })
                    successful_transcriptions += 1
                else:
                    logger.warning(f"❌ 片段 {i+1} (说话人: {diarized_segments[i]['speaker']}) 被 RMS 过滤或转录为空，已丢弃。")
                self.clear_gpu_memory()
            
            if successful_transcriptions == 0:
                logger.warning("❌ 所有片段转录失败，回退到完整转录")
                return self._transcribe_full_audio(audio_np, audio_duration, language)
                
        except Exception as e:
            logger.error(f"❌ 模块化转录流程失败: {e}")
            logger.error(f"详细堆栈: {traceback.format_exc()}")
            return self._transcribe_full_audio(audio_np, audio_duration, language)
        
        return results

    def _transcribe_single_segment(self, audio_np: np.ndarray, segment: Dict, language: str) -> Tuple[Optional[str], float]:
        """转录单个音频片段，并返回 (转录文本, 置信度)"""
        try:
            start = segment['start']
            end = segment['end']
            segment_duration = end - start
            
            if segment_duration < 0.1:
                return None, -10.0
            
            # 使用增强后的音频进行转录
            audio_segment = audio_np[int(start * self.SAMPLE_RATE):int(end * self.SAMPLE_RATE)]
            if len(audio_segment) == 0:
                return None, -10.0
            
            transcription, confidence_score = self.transcribe_segment(audio_segment, segment_duration, language)
            self.clear_gpu_memory()
            return transcription, confidence_score
            
        except Exception as e:
            logger.warning(f"片段转录失败: {e}")
            self.clear_gpu_memory()
            return None, -10.0

    def _transcribe_full_audio(self, audio_np: np.ndarray, audio_duration: float, language: str) -> List[Dict]:
        """完整音频转录（回退方案）"""
        transcription, _ = self.transcribe_segment(audio_np, audio_duration, language)
        return [{
            "role": "未知",
            "speaker": "SPEAKER_00",
            "text": transcription,
            "start": 0.0,
            "end": audio_duration
        }]
