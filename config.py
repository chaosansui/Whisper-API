import os
import torch

# 模型路径
MODEL_PATH = "/mnt/data/models/audio/whisper-large-v3"

# 性能优化配置
OPTIMIZATION_CONFIG = {
    "use_flash_attention": False,
    "chunk_length": 30,
    "batch_size": 1,
    "max_retries": 5,
    "timeout": 60.0,
    "n_mels": 256,
    "sample_rate": 16000
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# API 配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,
    "timeout_keep_alive": 300,
    "log_level": "info"
}