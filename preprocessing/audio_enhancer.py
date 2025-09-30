import numpy as np
import logging
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

class AudioEnhancer:
    """
    负责音频质量检查、降噪和信号增强。
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
        self.high_pass_freq = 80.0 
        self.low_pass_freq = 7000.0 

    def quality_check(self, audio_np: np.ndarray):
        
        rms = np.sqrt(np.mean(audio_np**2))
        if rms < 0.005:
            logger.warning(f"⚠️ 音频质量警告: RMS幅度极低 ({rms:.4f})，可能为静音或极低音量。")

    def denoise_and_enhance(self, audio_np: np.ndarray) -> np.ndarray:
        
        enhanced_audio_np = audio_np 
        
        logger.info("🎶 正在执行降噪和音频增强...")
        try:
            
            
            nyquist = 0.5 * self.sample_rate
            
            
            b, a = butter(2, self.high_pass_freq / nyquist, btype='high')
            processed_audio = filtfilt(b, a, audio_np)
            
            
            b, a = butter(2, self.low_pass_freq / nyquist, btype='low')
            processed_audio = filtfilt(b, a, processed_audio)

            
            max_abs = np.max(np.abs(processed_audio))
            if max_abs > 1e-6:
                
                enhanced_audio_np = processed_audio / max_abs * 0.95
            else:
                
                enhanced_audio_np = processed_audio

            logger.info("✅ 降噪和增强完成。")

        except Exception as e:
            logger.error(f"❌ 降噪和增强失败: {e}")
            logger.warning("⚠️ 降噪失败，已安全回退到使用原始音频进行后续处理。")
        
        return enhanced_audio_np