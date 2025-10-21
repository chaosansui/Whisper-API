import numpy as np
import logging
import resampy
import io
import soundfile as sf
import torch
from silero_vad import (
    load_silero_vad, 
    get_speech_timestamps, 
    save_audio, 
    read_audio, 
    VADIterator, 
    collect_chunks
)
from config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.TARGET_SAMPLE_RATE = 16000  # Whisper所需采样率
        self.vad_model = load_silero_vad(onnx=True)
        logger.info("✅ Silero VAD 模型已加载")
    
    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """基础音频预处理：读取、格式转换、重采样到16kHz"""
        try:
            # 读取音频
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            logger.info(f"原始音频: 形状={audio_np.shape}, 采样率={sr}, 时长={len(audio_np)/sr:.2f}秒")
            
            # 多声道处理
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
                logger.info(f"多声道合并: {audio_np.shape}")
            
            # 数据类型转换
            audio_np = audio_np.astype(np.float32)
            
            # 重采样到16kHz（Whisper要求）
            if sr != self.TARGET_SAMPLE_RATE:
                audio_np = resampy.resample(audio_np, sr, self.TARGET_SAMPLE_RATE)
                logger.info(f"重采样到16kHz: {sr}Hz → {self.TARGET_SAMPLE_RATE}Hz")
            
            return np.ascontiguousarray(audio_np)
            
        except Exception as e:
            logger.error(f"音频预处理失败: {str(e)}")
            raise ValueError(f"音频预处理失败: {str(e)}")

    def vad_segmentation(self, audio_np: np.ndarray) -> list:
        """使用 Silero VAD 进行语音活动检测和分割"""
        try:
            audio_np = np.ascontiguousarray(audio_np)
            
            # 获取 VAD 参数
            vad_params = self._get_vad_params()
            
            logger.info(f"VAD处理 - 输入音频: {len(audio_np)}样本, {len(audio_np)/self.TARGET_SAMPLE_RATE:.2f}秒")
            
            # VAD 检测（在16kHz下）
            audio_tensor = torch.from_numpy(audio_np.copy()).float()
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=self.TARGET_SAMPLE_RATE,
                **vad_params,
                return_seconds=False
            )
            
            if not speech_timestamps:
                logger.warning("VAD未检测到语音片段")
                return []
            
            logger.info(f"VAD检测到 {len(speech_timestamps)} 个语音片段")
            
            # 提取语音片段
            speech_segments = self._extract_speech_segments(audio_np, speech_timestamps)
            
            logger.info(f"VAD分割完成: {len(speech_segments)} 个有效语音片段")
            return speech_segments
            
        except Exception as e:
            logger.error(f"VAD处理失败: {str(e)}")
            return []

    def _get_vad_params(self) -> dict:
        """获取 VAD 参数配置"""
        return {
            "threshold": 0.2,        
            "min_silence_duration_ms": 100,   
            "min_speech_duration_ms": 200,    
            "speech_pad_ms": 150              
        }

    def _extract_speech_segments(self, audio_np: np.ndarray, speech_timestamps: list) -> list:
        """从时间戳提取语音片段"""
        speech_segments = []
        min_segment_duration = 0.5 * self.TARGET_SAMPLE_RATE  # 最小0.5秒
        max_segment_duration = 30 * self.TARGET_SAMPLE_RATE   # 最大30秒
        
        for i, segment in enumerate(speech_timestamps):
            start, end = segment['start'], segment['end']
            duration = end - start
            
            logger.info(f"片段 {i+1}: 开始={start}, 结束={end}, 样本数={duration}, 时长={duration/self.TARGET_SAMPLE_RATE:.2f}秒")
            
            # 检查片段是否太短
            if duration < min_segment_duration:
                logger.warning(f"片段过短 ({duration/self.TARGET_SAMPLE_RATE:.2f}秒 < 0.5秒)，跳过")
                continue
            
            # 处理过长片段
            if duration > max_segment_duration:
                split_segments = self._split_long_segment(audio_np, start, end, max_segment_duration)
                speech_segments.extend(split_segments)
                logger.info(f"长片段分割为 {len(split_segments)} 部分")
            else:
                segment_data = audio_np[start:end]
                if len(segment_data) > 0:
                    speech_segments.append(np.ascontiguousarray(segment_data))
                    logger.info(f"提取片段 {i+1}: 样本数={len(segment_data)}, 时长={len(segment_data)/self.TARGET_SAMPLE_RATE:.2f}秒")
        
        return speech_segments

    def _split_long_segment(self, audio_np: np.ndarray, start: int, end: int, 
                           max_duration: int) -> list:
        """分割过长的语音片段"""
        segments = []
        current_start = start
        
        while current_start < end:
            current_end = min(current_start + max_duration, end)
            segment_data = audio_np[current_start:current_end]
            
            if len(segment_data) > 0:
                segments.append(np.ascontiguousarray(segment_data))
            
            current_start = current_end
        
        return segments

    def merge_short_segments(self, segments: list, min_duration: float = 2.0) -> list:
        """合并过短的语音片段"""
        if not segments:
            return []
            
        merged_segments = []
        current_segment = segments[0]
        min_samples = int(min_duration * self.TARGET_SAMPLE_RATE)
        
        for i in range(1, len(segments)):
            # 如果当前片段很短，尝试与下一个合并
            if len(current_segment) < min_samples and i < len(segments):
                # 简单合并（实际应用中可能需要添加静音间隔）
                current_segment = np.concatenate([current_segment, segments[i]])
                logger.info(f"合并片段 {i} 和 {i+1}")
            else:
                merged_segments.append(current_segment)
                current_segment = segments[i]
        
        # 添加最后一个片段
        if len(current_segment) > 0:
            merged_segments.append(current_segment)
        
        logger.info(f"片段合并完成: {len(segments)} → {len(merged_segments)} 个片段")
        return merged_segments

    def validate_audio(self, audio_np: np.ndarray) -> bool:
        """验证音频质量"""
        try:
            # 检查音频长度
            if len(audio_np) < self.TARGET_SAMPLE_RATE * 0.5:  # 至少0.5秒
                logger.warning("音频过短")
                return False
            
            # 检查音量
            volume = np.sqrt(np.mean(audio_np**2))
            if volume < 0.001:  # 音量阈值
                logger.warning("音频音量过低")
                return False
            
            logger.info("✅ 音频验证通过")
            return True
            
        except Exception as e:
            logger.warning(f"音频验证失败: {e}")
            return False

    def get_audio_info(self, audio_np: np.ndarray) -> dict:
        """获取音频信息"""
        duration = len(audio_np) / self.TARGET_SAMPLE_RATE
        volume = np.sqrt(np.mean(audio_np**2))
        
        return {
            "duration_seconds": round(duration, 2),
            "samples": len(audio_np),
            "sample_rate": self.TARGET_SAMPLE_RATE,
            "volume": round(volume, 4),
            "max_amplitude": round(np.max(np.abs(audio_np)), 4)
        }

    def clear_resources(self):
        """清理资源"""
        if hasattr(self, 'vad_model'):
            del self.vad_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ 音频处理器资源已清理")