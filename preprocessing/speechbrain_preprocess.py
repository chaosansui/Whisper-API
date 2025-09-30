import numpy as np
import logging
import torch
from config import SAMPLE_RATE, SPEAKER_EMBEDDING_PATH
from preprocessing.audio_enhancer import AudioEnhancer
from preprocessing.vad_segmenter import VADSegmenter, VAD_SAMPLE_RATE
from preprocessing.diarization_core import DiarizationCore

logger = logging.getLogger(__name__)

def run_diarization_pipeline(audio_file_path: str):
    """执行完整的 Diarization 流程"""
    logger.info("--- 开始语音处理流程 ---")
    

    device = "cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else "cuda:0" if torch.cuda.is_available() else "cpu"
    
    audio_np = np.random.rand(SAMPLE_RATE * 10).astype(np.float32) 
    
    enhancer = AudioEnhancer(sample_rate=SAMPLE_RATE)
    enhancer.quality_check(audio_np)
    enhanced_audio_np = enhancer.denoise_and_enhance(audio_np)
    
    if SAMPLE_RATE != VAD_SAMPLE_RATE:
        logger.warning(f"音频需要重采样到 {VAD_SAMPLE_RATE} 才能进行 VAD...")
        enhanced_audio_resampled_np = enhanced_audio_np 
    else:
        enhanced_audio_resampled_np = enhanced_audio_np

    vad_segmenter = VADSegmenter(sample_rate=VAD_SAMPLE_RATE, device='cpu')
    vad_segments = vad_segmenter.get_speech_segments(enhanced_audio_resampled_np)


    diarization_core = DiarizationCore(
        speaker_embedding_path=SPEAKER_EMBEDDING_PATH,
        sample_rate=SAMPLE_RATE,
        device=device
    )

    diarized_segments = diarization_core.diarize(enhanced_audio_np, vad_segments) 
    
    logger.info("--- Diarization 流程结束 ---")
    return diarized_segments
