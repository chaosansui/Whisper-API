import numpy as np
import logging
import resampy
import io
import soundfile as sf
import torch
from silero_vad import (
    load_silero_vad, 
    get_speech_timestamps
)
from config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.TARGET_SAMPLE_RATE = 16000  
        self.vad_model = load_silero_vad(onnx=True)
        logger.info("✅ Silero VAD 模型已加载")
    
    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """基础音频预处理：读取、格式转换、重采样到16kHz"""
        try:
            # 尝试直接读取
            try:
                audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                # 某些格式 soundfile 不支持，这里可以考虑降级方案或者直接抛出更明确的错误
                logger.error(f"SoundFile读取失败: {e}，请确保上传的是标准音频格式")
                raise ValueError("不支持的音频格式或文件已损坏")

            logger.info(f"原始音频: 形状={audio_np.shape}, 采样率={sr}, 时长={len(audio_np)/sr:.2f}秒")
            
            # 多声道处理 (转单声道)
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
                logger.info(f"多声道合并: {audio_np.shape}")
            
            # 数据类型转换
            audio_np = audio_np.astype(np.float32)
            
            # 重采样
            if sr != self.TARGET_SAMPLE_RATE:
                # resampy 质量好但速度稍慢，对于长音频可以接受
                audio_np = resampy.resample(audio_np, sr, self.TARGET_SAMPLE_RATE)
                logger.info(f"重采样到16kHz: {sr}Hz → {self.TARGET_SAMPLE_RATE}Hz")
            
            # 确保内存连续 (Silero VAD 要求)
            return np.ascontiguousarray(audio_np)
            
        except Exception as e:
            logger.error(f"音频预处理失败: {str(e)}")
            raise ValueError(f"音频预处理失败: {str(e)}")

    def vad_segmentation(self, audio_np: np.ndarray) -> list:
        """使用 Silero VAD 进行语音活动检测和分割"""
        try:
            # 再次确保连续性
            if not audio_np.flags['C_CONTIGUOUS']:
                audio_np = np.ascontiguousarray(audio_np)
            
            # 获取 VAD 参数
            vad_params = self._get_vad_params()
            
            logger.info(f"VAD处理 - 输入音频: {len(audio_np)}样本, {len(audio_np)/self.TARGET_SAMPLE_RATE:.2f}秒")
            
            # VAD 检测（Silero 需要 Tensor 输入）
            # 注意：Silero VAD 在 CPU 上跑通常就够快了，转 GPU 反而有拷贝开销
            audio_tensor = torch.from_numpy(audio_np.copy()).float()
            
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=self.TARGET_SAMPLE_RATE,
                **vad_params,
                return_seconds=False
            )
            
            if not speech_timestamps:
                # 如果没有检测到明显的语音，返回空列表
                # 外层 WhisperASR 会有兜底逻辑（直接处理整个音频）
                logger.warning("VAD未检测到语音片段")
                return []
            
            logger.info(f"VAD检测到 {len(speech_timestamps)} 个语音片段")
            
            # 提取语音片段
            speech_segments = self._extract_speech_segments(audio_np, speech_timestamps)
            
            logger.info(f"VAD分割完成: {len(speech_segments)} 个有效语音片段")
            return speech_segments
            
        except Exception as e:
            logger.error(f"VAD处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果 VAD 崩了，返回空，让外层兜底
            return []

    def _get_vad_params(self) -> dict:
        """
        VAD 参数调优
        """
        return {
            "threshold": 0.35,        # 提高阈值 (原0.2)，减少呼吸声/底噪被误判为语音
            "min_silence_duration_ms": 100, # 保持默认，允许较短的停顿
            "min_speech_duration_ms": 250,  # 至少 0.25秒才算一句话，防止电流声干扰
            "speech_pad_ms": 200            # 前后各保留 0.2s，防止吞首尾音
        }

    def _extract_speech_segments(self, audio_np: np.ndarray, speech_timestamps: list) -> list:
        """从时间戳提取语音片段"""
        speech_segments = []
        # 允许稍短的片段，交给后续合并逻辑处理
        min_segment_duration = 0.2 * self.TARGET_SAMPLE_RATE 
        
        for i, segment in enumerate(speech_timestamps):
            start, end = segment['start'], segment['end']
            duration = end - start
            
            # 过滤极短片段
            if duration < min_segment_duration:
                continue
            
            segment_data = audio_np[start:end]
            if len(segment_data) > 0:
                speech_segments.append(np.ascontiguousarray(segment_data))
        
        return speech_segments

    # 已删除 _split_long_segment，因为 WhisperASR 那边会强制 Padding 和截断/合并，
    # 这里不需要复杂的切分逻辑。Silero 切出来的通常都是自然的句子。

    def validate_audio(self, audio_np: np.ndarray) -> bool:
        """验证音频质量"""
        try:
            # 长度检查
            if len(audio_np) < 1600:  # 0.1秒
                logger.warning("音频极短")
                return False
            
            # 音量检查
            # 计算 RMS (均方根) 能量
            rms = np.sqrt(np.mean(audio_np**2))
            if rms < 0.0001:  # 极低音量阈值
                logger.warning(f"音频接近纯静音 (RMS: {rms:.6f})")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"音频验证失败: {e}")
            return False

    def clear_resources(self):
        """清理资源"""
        if hasattr(self, 'vad_model'):
            del self.vad_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ 音频处理器资源已清理")