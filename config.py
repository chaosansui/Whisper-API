import os
import torch
import logging
from typing import Dict, Any

class Config:
    """简化配置管理"""
    
    # 路径配置
    MODEL_PATH = "/mnt/data/models/audio/whisper-large-v3"
    CACHE_DIR = "./cache"
    
    # 设备配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # API 配置
    API = {
        "host": "0.0.0.0",
        "port": 8008,
        "timeout_keep_alive": 300,
        "log_level": "info",
        "max_file_size": 100 * 1024 * 1024
    }
    
    # 日志配置
    LOGGING = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
    
    @classmethod
    def setup_logging(cls):
        """设置日志"""
        logging.basicConfig(
            level=cls.LOGGING["level"],
            format=cls.LOGGING["format"]
        )
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """获取设备信息"""
        info = {"device": str(cls.DEVICE)}
        if cls.DEVICE.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            })
        return info

Config.setup_logging()