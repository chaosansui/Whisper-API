import torch
import numpy as np
import logging
import traceback
from typing import List, Dict
from config import VAD_SAMPLE_RATE
logger = logging.getLogger(__name__)

class VADSegmenter:
    """负责运行 Silero VAD 并执行边界填充 (Padding)。"""
    def __init__(self, sample_rate: int, device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.device = device
        self.vad_model = None
        self.get_timestamps_func = None
        self._load_vad_model()

    def _load_vad_model(self):
        try:
            logger.info("🔄 尝试通过 PyTorch Hub 加载 Silero VAD 模型...")
            model, utils = torch.hub.load(
                 repo_or_dir='snakers4/silero-vad',
                 model='silero_vad',
                 force_reload=False,
            )
            (self.get_timestamps_func, _, _, _, _) = utils
            self.vad_model = model.eval().to(self.device)
            logger.info(f"✅ Silero VAD 模型加载成功 ({self.device})")
        except Exception as e:
            logger.error(f"❌ Silero VAD 模型加载失败: {e}")
            raise

    def get_speech_segments(self, audio_np: np.ndarray) -> List[Dict]:

        audio_duration = len(audio_np) / self.sample_rate
        
        # 1. 核心 VAD 检测
        audio_tensor = torch.from_numpy(audio_np).float() 
        segments = []
        try:
            # 优化参数：提高敏感度，捕捉短语
            segments = self.get_timestamps_func(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=VAD_SAMPLE_RATE, # Silero VAD 默认使用 16000
                threshold=0.30, 
                min_speech_duration_ms=0.15,
                min_silence_duration_ms=0.3 
            )
        except Exception as e:
            logger.error(f"❌ Silero VAD 内部调用失败: {e}")
            # VAD 失败时，使用 10 秒固定长度分块回退 (TODO: 替换为实际的回退逻辑)
            segments = self._create_fixed_segments(audio_duration, chunk_size=10.0)

        # 2. VAD Segment Padding
        START_PADDING = 0.10 
        END_PADDING = 0.20   
        padded_segments = []
        for ts in segments:
            start_sec = ts['start'] / VAD_SAMPLE_RATE
            end_sec = ts['end'] / VAD_SAMPLE_RATE

            new_start = max(0.0, start_sec - START_PADDING)
            new_end = min(audio_duration, end_sec + END_PADDING)
            
            if new_end - new_start >= 0.2:
                padded_segments.append({
                    "start": new_start,
                    "end": new_end,
                    "speaker": None 
                })
        
        logger.info(f"🎯 VAD Segmenter 输出 {len(padded_segments)} 个片段")
        return padded_segments

    def _create_fixed_segments(self, duration: float, chunk_size: float = 10.0) -> List[Dict]:
        """作为 VAD 失败的回退机制"""
        segments = []
        start = 0.0
        while start < duration:
            end = min(start + chunk_size, duration)
            if end - start >= 0.1:
                # 转换回 VAD 采样点，适配 VAD 内部输出格式
                segments.append({
                    "start": int(start * VAD_SAMPLE_RATE),
                    "end": int(end * VAD_SAMPLE_RATE),
                })
            start = end
        return segments
