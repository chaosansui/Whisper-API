import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import io
import soundfile as sf
import numpy as np
import resampy
import traceback
import re
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from silero_vad import (
    load_silero_vad, get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks
)
import torch.nn as nn
from omegaconf import OmegaConf
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI 配置
MODEL_PATH = "/mnt/data/models/audio/whisper-large-v3"

# 定义响应模型
class AudioResponse(BaseModel):
    transcription: str
    language: str = "auto"
    detected_language: str = "unknown"

# Whisper ASR 类
class WhisperASR:
    def __init__(self):
        """初始化 WhisperASR"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        logger.info(f"使用设备: {self.device}, GPU: {torch.cuda.get_device_name(0)}")

        self.whisper_model = None
        self.processor = None
        self.vad_model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        self.SAMPLE_RATE = 16000
        
        # 性能优化配置
        self.optimization_config = {
            "use_flash_attention": False,
            "chunk_length": 30,
            "batch_size": 1,
            "max_retries": 5,
            "timeout": 60.0,
            "n_mels": 256 
        }
        
        # 扩展语言映射表
        self.language_mapping = {
            "zh": "zh",
            "yue": "yue",
            "yue_Hant": "yue",
            "yue_Hans": "yue",
            "en": "en",
            "cn": "zh",
            "zh-tw": "zh",
            "zh-cn": "zh",
            "ja": "ja",
            "ko": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
            "vi": "vi",
            "chinese": "zh",
            "cantonese": "yue",
            "english": "en"
        }
        
        # 支持的语言列表
        self.supported_languages = [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it',
            'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur',
            'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn',
            'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si',
            'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo',
            'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha',
            'ba', 'jw', 'su', 'yue'
        ]
        
        # 粤语专用词汇表
        self.cantonese_corrections = {
            "是不是": "係唔係",
            "这样": "咁樣",
            "那个": "嗰個",
            "这里": "呢度",
            "什么时候": "幾時",
            "为什么": "點解",
            "很好": "好好",
            "没有": "冇",
            "的": "嘅",
            "他": "佢",
            "我们": "我哋"
        }
        
        # 英文专用词汇表
        self.english_corrections = {
            "there": "their",
            "your": "you're",
            "its": "it's",
            "to": "too",
            "then": "than",
            "weather": "whether",
            "accept": "except",
            "OK": "okay",
            "Punky": "Funky",
            "Punky Finance": "Funky Finance"
        }

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

    def load_models(self, model_path):
        """加载 Whisper 和 VAD 模型"""
        try:
            self.clear_gpu_memory()
            logger.info("🔄 加载 Whisper 模型...")
            self.processor = WhisperProcessor.from_pretrained(model_path, num_mel_bins=self.optimization_config["n_mels"])
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
            t = np.linspace(0, 1, 16000)
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
                for lang in ["yue", "en", "zh"]:
                    _ = self.whisper_model.generate(
                        input_features,
                        task="transcribe",
                        max_length=10,
                        num_beams=1,
                        language=lang
                    )
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}, 继续启动")

    def detect_language(self, audio_np, segment_duration=5.0, retries=3):
        """优化语言检测，支持粤语+英文混合"""
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
                generated_ids = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                    max_new_tokens=10,
                    return_dict_in_generate=True
                )
                lang_tokens = generated_ids.sequences[0]
                detected_languages = []
                for token in lang_tokens:
                    lang_code = self.processor.tokenizer.convert_ids_to_tokens([token])[0]
                    lang_code = re.sub(r'[<>]', '', lang_code).strip().lower()
                    if lang_code in self.supported_languages:
                        detected_languages.append((lang_code, 0.5))  # 默认概率
                logger.info(f"语言检测结果: {detected_languages}")
                for lang_code, prob in detected_languages:
                    if lang_code in ['yue', 'yue_Hant', 'yue_Hans'] and prob > 0.03:
                        logger.info(f"检测到粤语，概率: {prob:.3f}")
                        return self.language_mapping.get(lang_code, lang_code)
                    if lang_code == 'en' and prob > 0.05:
                        logger.info(f"检测到英文，概率: {prob:.3f}")
                        return self.language_mapping.get(lang_code, lang_code)
                for lang_code, prob in detected_languages:
                    if lang_code in ['zh', 'zh-cn', 'zh-tw', 'cn'] and prob > 0.08:
                        return self.language_mapping.get(lang_code, lang_code)
                if retries > 0:
                    logger.warning(f"语言检测结果为空，剩余重试次数: {retries}，尝试更短片段")
                    return self.detect_language(audio_np, segment_duration=segment_duration/2, retries=retries-1)
                logger.warning("未检测到有效语言，回退到粤语")
                return "yue"
        except Exception as e:
            logger.warning(f"语言检测失败: {e}，剩余重试次数: {retries}")
            if retries > 0:
                return self.detect_language(audio_np, segment_duration=segment_duration/2, retries=retries-1)
            logger.warning("语言检测失败，回退到粤语")
            return "yue"

    def estimate_audio_quality(self, audio_np):
        """估计音频质量，优化信噪比计算"""
        try:
            freqs, psd = welch(audio_np, fs=self.SAMPLE_RATE, nperseg=1024)
            signal_power = np.mean(psd[(freqs >= 150) & (freqs <= 4500)])
            noise_power = np.mean(psd[(freqs < 100) | (freqs > 5000)])
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                quality = min(1.0, max(0.0, (snr - 5) / 30))
            else:
                quality = 0.8
            logger.info(f"音频质量估计: SNR={snr:.2f}dB, 质量={quality:.2f}")
            return quality
        except Exception as e:
            logger.warning(f"音频质量估计失败: {e}, 使用默认质量 0.7")
            return 0.7

    def process_audio(self, audio_bytes):
        """处理音频数据"""
        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            duration = len(audio_np) / sr
            logger.info(f"原始音频: 形状={audio_np.shape}, 采样率={sr}, 时长={duration:.2f}秒")
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
                logger.info(f"通道平均后: {audio_np.shape}")
            audio_np = audio_np.astype(np.float32)
            if sr != self.SAMPLE_RATE:
                audio_np = resampy.resample(audio_np, sr, self.SAMPLE_RATE)
                logger.info(f"重采样后: {audio_np.shape}, 采样率={self.SAMPLE_RATE}")
            return audio_np
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
            raise ValueError(f"音频处理失败: {str(e)}")

    def enhance_telephone_audio(self, audio_np):
        """增强电话录音质量，优化粤语和英文，添加降噪"""
        try:
            # 动态光谱减法降噪
            freqs, psd = welch(audio_np, fs=self.SAMPLE_RATE, nperseg=1024)
            noise_psd = np.mean(psd[(freqs < 100) | (freqs > 5000)])
            noise_threshold = np.sqrt(noise_psd) * 0.9
            audio_np[np.abs(audio_np) < noise_threshold] *= 0.1

            # 带通滤波
            nyquist = self.SAMPLE_RATE / 2
            lowcut = 100 / nyquist
            highcut = 5000 / nyquist
            b, a = signal.cheby2(8, 60, [lowcut, highcut], btype='bandpass')
            audio_np = signal.filtfilt(b, a, audio_np)
            
            # 自适应增益控制
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val
                compressed = np.zeros_like(audio_np)
                low_mask = np.abs(audio_np) < 0.06
                compressed[low_mask] = audio_np[low_mask] * 3.0
                mid_mask = (np.abs(audio_np) >= 0.06) & (np.abs(audio_np) < 0.3)
                compressed[mid_mask] = audio_np[mid_mask] * 1.8
                high_mask = np.abs(audio_np) >= 0.3
                compressed[high_mask] = audio_np[high_mask] * 0.7
                audio_np = compressed
            
            # 语音增强
            b_eq, a_eq = signal.butter(4, [600/nyquist, 3200/nyquist], btype='bandpass')
            boosted = signal.lfilter(b_eq, a_eq, audio_np)
            audio_np = audio_np * 0.5 + boosted * 0.5
            
            # 平滑处理
            audio_np = uniform_filter1d(audio_np, size=5)
            logger.info("✅ 高级电话录音增强完成")
            return audio_np.astype(np.float32)
        except Exception as e:
            logger.warning(f"高级音频增强失败: {e}, 使用基础增强")
            return self.basic_enhance_telephone_audio(audio_np)

    def basic_enhance_telephone_audio(self, audio_np):
        """基础电话录音增强"""
        nyquist = self.SAMPLE_RATE / 2
        lowcut = 100 / nyquist
        highcut = 5000 / nyquist
        b, a = signal.butter(4, [lowcut, highcut], btype='band')
        audio_np = signal.filtfilt(b, a, audio_np)
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.9
        return audio_np

    def optimize_vad_parameters(self, audio_tensor):
        """优化VAD参数，针对粤语和英文"""
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
        """自适应VAD处理，强制分段"""
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
                {
                    "threshold": 0.008,
                    "min_silence_duration_ms": 30,
                    "min_speech_duration_ms": 50,
                    "speech_pad_ms": 10
                },
                {
                    "threshold": 0.01,
                    "min_silence_duration_ms": 20,
                    "min_speech_duration_ms": 30,
                    "speech_pad_ms": 5
                },
                {
                    "threshold": 0.005,
                    "min_silence_duration_ms": 10,
                    "min_speech_duration_ms": 20,
                    "speech_pad_ms": 5
                },
                {
                    "threshold": 0.001,
                    "min_silence_duration_ms": 10,
                    "min_speech_duration_ms": 10,
                    "speech_pad_ms": 5
                }
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

    def optimize_generation_parameters(self, audio_duration, audio_quality=0.5, language=None):
        """优化生成参数，支持智能客服场景"""
        base_params = {
            "max_length": int(audio_duration * 60),
            "num_beams": 12,
            "temperature": 0.2,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "repetition_penalty": 1.2
        }
        if language == "yue":
            base_params.update({
                "temperature": 0.15,
                "num_beams": 12,
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 4
            })
        elif language == "en":
            base_params.update({
                "temperature": 0.15,
                "num_beams": 12,
                "repetition_penalty": 1.25,
                "no_repeat_ngram_size": 3
            })
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
        if audio_quality < 0.4:
            base_params.update({
                "num_beams": 14,
                "repetition_penalty": 1.5
            })
        return base_params

    def clean_text(self, text, language=None):
        """清理转录文本，支持粤语和英文"""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:，。！？；：\-()（）]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.!?。！？])([^\s])', r'\1 \2', text)
        # 修复常见错误
        text = re.sub(r'K2670666\b', 'K26706656', text)
        if language == "yue":
            text = self.postprocess_cantonese(text)
        elif language == "en":
            text = self.postprocess_english(text)
        return text

    def postprocess_cantonese(self, text):
        """粤语转录后处理"""
        if not text:
            return text
        for wrong, correct in self.cantonese_corrections.items():
            text = text.replace(wrong, correct)
        text = re.sub(r'(唔)([^係要好錯])', r'\1 \2', text)
        text = re.sub(r'(嘅)([^\.。!！?？,，])', r'\1 \2', text)
        return text

    def postprocess_english(self, text):
        """英文转录后处理"""
        if not text:
            return text
        for wrong, correct in self.english_corrections.items():
            text = text.replace(wrong, correct)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        text = text[0].upper() + text[1:] if text else text
        return text

    def transcribe_cantonese_optimized(self, audio_np, audio_duration, audio_quality=0.5):
        """粤语优化转录"""
        logger.info("使用粤语优化转录模式")
        cantonese_params = {
            "max_length": int(audio_duration * 60),
            "num_beams": 12,
            "temperature": 0.15,
            "no_repeat_ngram_size": 4,
            "early_stopping": False,
            "repetition_penalty": 1.3,
            "language": "yue",
            "task": "transcribe"
        }
        inputs = self.processor(
            audio_np,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        model_dtype = next(self.whisper_model.parameters()).dtype
        input_features = input_features.to(model_dtype)
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                input_features,
                **cantonese_params
            )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_transcription = self.clean_text(transcription, "yue")
        logger.info(f"粤语优化转录结果: {final_transcription}")
        return final_transcription

    def transcribe_english_optimized(self, audio_np, audio_duration, audio_quality=0.5):
        """英文优化转录"""
        logger.info("使用英文优化转录模式")
        english_params = {
            "max_length": int(audio_duration * 60),
            "num_beams": 12,
            "temperature": 0.15,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "repetition_penalty": 1.25,
            "language": "en",
            "task": "transcribe"
        }
        inputs = self.processor(
            audio_np,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        model_dtype = next(self.whisper_model.parameters()).dtype
        input_features = input_features.to(model_dtype)
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                input_features,
                **english_params
            )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_transcription = self.clean_text(transcription, "en")
        logger.info(f"英文优化转录结果: {final_transcription}")
        return final_transcription

    def transcribe_mixed_audio(self, audio_np, audio_duration, audio_quality=0.5):
        """处理粤语+英文混合音频，移除语言标签"""
        logger.info("处理粤语+英文混合音频，移除语言标签")
        audio_tensor = torch.from_numpy(audio_np).float().to("cpu")
        if len(audio_tensor.shape) > 1:
            audio_tensor = torch.mean(audio_tensor, dim=1)
        speech_timestamps = self.adaptive_vad_processing(audio_tensor)
        if not speech_timestamps:
            logger.warning("VAD 未检测到语音，尝试完整音频转录")
            detected_language = self.detect_language(audio_np)
            if detected_language == "yue":
                return self.transcribe_cantonese_optimized(audio_np, audio_duration, audio_quality), detected_language
            elif detected_language == "en":
                return self.transcribe_english_optimized(audio_np, audio_duration, audio_quality), detected_language
            else:
                return self.transcribe_full_audio(audio_np, audio_duration, detected_language, audio_quality), detected_language
        all_transcriptions = []
        for i, segment_info in enumerate(speech_timestamps):
            start_sample = segment_info['start']
            end_sample = segment_info['end']
            segment = audio_np[start_sample:end_sample]
            segment_duration = (end_sample - start_sample) / self.SAMPLE_RATE
            logger.info(f"处理片段 {i+1}/{len(speech_timestamps)}: 时长: {segment_duration:.2f}秒")
            segment_language = self.detect_language(segment, segment_duration=min(1.0, segment_duration))
            if segment_language == "yue":
                transcription = self.transcribe_cantonese_optimized(segment, segment_duration, audio_quality)
            elif segment_language == "en":
                transcription = self.transcribe_english_optimized(segment, segment_duration, audio_quality)
            else:
                transcription = self.transcribe_full_audio(segment, segment_duration, segment_language, audio_quality)
            if transcription:
                all_transcriptions.append(transcription)
                logger.info(f"片段 {i+1} 转录结果: {transcription} ({segment_language})")
            torch.cuda.empty_cache()
        final_transcription = " ".join(all_transcriptions)
        logger.info("已移除语言标签")
        return final_transcription, "mixed"

    def transcribe_full_audio(self, audio_np, audio_duration, detected_language=None, audio_quality=0.5):
        """直接处理完整音频"""
        if detected_language == "yue":
            return self.transcribe_cantonese_optimized(audio_np, audio_duration, audio_quality)
        elif detected_language == "en":
            return self.transcribe_english_optimized(audio_np, audio_duration, audio_quality)
        logger.info("🔄 直接处理完整音频...")
        # 转换语言代码
        if detected_language:
            detected_language = self.language_mapping.get(detected_language, "yue")
        if detected_language not in self.supported_languages:
            logger.warning(f"无效语言代码: {detected_language}，回退到粤语")
            detected_language = "yue"
        # 分块处理长音频
        chunk_duration = 30  # 每块30秒
        num_chunks = int(np.ceil(audio_duration / chunk_duration))
        all_transcriptions = []
        for i in range(num_chunks):
            start_sample = int(i * chunk_duration * self.SAMPLE_RATE)
            end_sample = min(int((i + 1) * chunk_duration * self.SAMPLE_RATE), len(audio_np))
            chunk = audio_np[start_sample:end_sample]
            chunk_duration = (end_sample - start_sample) / self.SAMPLE_RATE
            logger.info(f"处理音频块 {i+1}/{num_chunks}: 时长: {chunk_duration:.2f}秒")
            gen_params = self.optimize_generation_parameters(chunk_duration, audio_quality, detected_language)
            inputs = self.processor(
                chunk,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(self.device)
            model_dtype = next(self.whisper_model.parameters()).dtype
            input_features = input_features.to(model_dtype)
            with torch.no_grad():
                generate_kwargs = gen_params.copy()
                if detected_language:
                    generate_kwargs["language"] = detected_language
                try:
                    generated_ids = self.whisper_model.generate(
                        input_features,
                        task="transcribe",
                        **generate_kwargs
                    )
                except ValueError as e:
                    logger.warning(f"生成失败: {e}，尝试无语言指定转录")
                    generate_kwargs.pop("language", None)
                    generated_ids = self.whisper_model.generate(
                        input_features,
                        task="transcribe",
                        **generate_kwargs
                    )
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            final_transcription = self.clean_text(transcription, detected_language)
            if final_transcription:
                all_transcriptions.append(final_transcription)
            torch.cuda.empty_cache()
        final_transcription = " ".join(all_transcriptions)
        logger.info(f"完整音频转录结果: {final_transcription}")
        return final_transcription

    def transcribe_audio(self, audio_bytes, forced_language=None):
        """转录完整音频，支持强制语言设置"""
        if not self.model_loaded:
            logger.error("模型未加载")
            return "模型未加载，请检查服务状态", "unknown"
        try:
            start_time = time.time()
            audio_np = self.process_audio(audio_bytes)
            audio_np = self.enhance_telephone_audio(audio_np)
            audio_duration = len(audio_np) / self.SAMPLE_RATE
            audio_quality = self.estimate_audio_quality(audio_np)
            logger.info(f"增强后音频时长: {audio_duration:.2f}秒, 质量: {audio_quality:.2f}")
            if forced_language:
                detected_language = self.language_mapping.get(forced_language, forced_language)
                logger.info(f"使用强制语言: {detected_language}")
            else:
                detected_language = self.detect_language(audio_np)
            if detected_language in ["yue", "en"]:
                transcription, final_language = self.transcribe_mixed_audio(audio_np, audio_duration, audio_quality)
            else:
                transcription = self.transcribe_full_audio(audio_np, audio_duration, detected_language, audio_quality)
                final_language = detected_language or "unknown"
            logger.info(f"转录总耗时: {time.time() - start_time:.2f}秒")
            return transcription if transcription else "未检测到语音", final_language
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            traceback.print_exc()
            return f"转录失败: {str(e)}", "error"

# 创建全局实例
whisperAsr = WhisperASR()

@asynccontextmanager
async def lifespan(app):
    """FastAPI 生命周期管理"""
    try:
        logger.info("🚀 启动服务，加载模型中...")
        whisperAsr.load_models(MODEL_PATH)
        logger.info("✅ 服务启动完成")
        yield
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    finally:
        logger.info("🛑 服务关闭，清理资源...")
        whisperAsr.clear_gpu_memory()

# 创建 FastAPI 应用
app = FastAPI(
    title="Whisper语音识别API",
    description="基于Whisper的多语言语音识别服务，优化智能客服场景（粤语+英文，含口音和噪音）",
    version="2.7.1",
    lifespan=lifespan
)

@app.post("/transcribe", response_model=AudioResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    forced_language: Optional[str] = Query(None, description="强制指定语言，如：cantonese, english")
):
    """音频转录接口，支持强制语言设置"""
    try:
        if not whisperAsr.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")
        start_time = time.time()
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        logger.info(f"收到音频文件: {file.filename}, 大小: {len(audio_bytes)} bytes")
        loop = asyncio.get_event_loop()
        transcription, detected_language = await loop.run_in_executor(
            whisperAsr.executor, whisperAsr.transcribe_audio, audio_bytes, forced_language
        )
        total_time = time.time() - start_time
        logger.info(f"请求处理完成，总耗时: {total_time:.2f} 秒")
        return AudioResponse(
            transcription=transcription, 
            language="auto",
            detected_language=detected_language
        )
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        whisperAsr.clear_gpu_memory()
        raise HTTPException(status_code=500, detail="GPU内存不足")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Whisper多语言语音识别服务 - 智能客服优化版",
        "status": "运行中" if whisperAsr.model_loaded else "启动中",
        "supported_languages": ["auto", "cantonese", "english", "chinese", "japanese", "korean", "vietnamese"],
        "endpoints": {
            "transcribe": "POST /transcribe - 上传音频进行转录",
            "health": "GET /health - 服务健康检查",
            "info": "GET /info - 模型信息"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if whisperAsr.model_loaded else "unhealthy",
        "model_loaded": whisperAsr.model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }

@app.get("/info")
async def model_info():
    """模型信息接口"""
    if not whisperAsr.model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {
        "model_name": "Whisper-large-v3",
        "language": "多语言自动检测，优化粤语和英文（智能客服场景）",
        "special_optimization": "粤语+英文混合，口音和噪音处理，长音频分块，无语言标签输出",
        "device": str(whisperAsr.device),
        "model_loaded": True,
        "optimizations": ["n_mels=256", "beam_size=10", "temperature=0.2", "动态降噪", "自适应VAD", "粤语优化", "英文优化", "混合语言处理", "长音频分块", "无语言标签"]
    }

if __name__ == "__main__":
    logger.info("Starting Whisper ASR Server with Cantonese+English optimization for customer service...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_keep_alive=300,
        log_level="info"
    )