import soundfile as sf
import numpy as np
import resampy
import logging
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from config import SAMPLE_RATE
import io

logger = logging.getLogger(__name__)

def process_audio(audio_bytes):
    """处理音频数据"""
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(audio_np) / sr
        logger.info(f"原始音频: 形状={audio_np.shape}, 采样率={sr}, 时长={duration:.2f}秒")
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)
            logger.info(f"通道平均后: {audio_np.shape}")
        audio_np = audio_np.astype(np.float32)
        if sr != SAMPLE_RATE:
            audio_np = resampy.resample(audio_np, sr, SAMPLE_RATE)
            logger.info(f"重采样后: {audio_np.shape}, 采样率={SAMPLE_RATE}")
        return audio_np
    except Exception as e:
        logger.error(f"音频处理失败: {str(e)}")
        raise ValueError(f"音频处理失败: {str(e)}")

def enhance_telephone_audio(audio_np):
    """增强电话录音质量"""
    try:
        freqs, psd = welch(audio_np, fs=SAMPLE_RATE, nperseg=1024)
        noise_psd = np.mean(psd[(freqs < 100) | (freqs > 5000)])
        noise_threshold = np.sqrt(noise_psd) * 0.9
        audio_np[np.abs(audio_np) < noise_threshold] *= 0.1

        nyquist = SAMPLE_RATE / 2
        lowcut = 100 / nyquist
        highcut = 5000 / nyquist
        b, a = signal.cheby2(8, 60, [lowcut, highcut], btype='bandpass')
        audio_np = signal.filtfilt(b, a, audio_np)
        
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
        
        b_eq, a_eq = signal.butter(4, [600/nyquist, 3200/nyquist], btype='bandpass')
        boosted = signal.lfilter(b_eq, a_eq, audio_np)
        audio_np = audio_np * 0.5 + boosted * 0.5
        
        audio_np = uniform_filter1d(audio_np, size=5)
        logger.info("✅ 电话录音增强完成")
        return audio_np.astype(np.float32)
    except Exception as e:
        logger.warning(f"音频增强失败: {e}, 使用基础增强")
        return basic_enhance_telephone_audio(audio_np)

def basic_enhance_telephone_audio(audio_np):
    """基础电话录音增强"""
    nyquist = SAMPLE_RATE / 2
    lowcut = 100 / nyquist
    highcut = 5000 / nyquist
    b, a = signal.butter(4, [lowcut, highcut], btype='band')
    audio_np = signal.filtfilt(b, a, audio_np)
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.9
    return audio_np

def estimate_audio_quality(audio_np):
    """估计音频质量（信噪比）"""
    try:
        freqs, psd = welch(audio_np, fs=SAMPLE_RATE, nperseg=1024)
        signal_power = np.mean(psd[(freqs >= 150) & (freqs <= 4500)])
        noise_power = np.mean(psd[(freqs < 100) | (freqs > 5000)])
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            quality = min(1.0, max(0.0, (snr - 5) / 30))
        else:
            quality = 0.8
        logger.info(f"音频质量: 信噪比={snr:.2f}dB, 质量={quality:.2f}")
        return quality
    except Exception as e:
        logger.warning(f"音频质量估计失败: {e}, 使用默认质量 0.7")
        return 0.7