import os
# 优化显存分配策略，防止碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
import asyncio
import uvicorn
import time
import setproctitle
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from whisper_asr import WhisperASR
from config import Config

# 设置进程名
setproctitle.setproctitle("whisper_server")

# 配置日志
logging.basicConfig(level=Config.LOGGING["level"], format=Config.LOGGING["format"])
logger = logging.getLogger(__name__)

# --- 数据模型 ---
class AudioResponse(BaseModel):
    transcription: str
    language: str = "auto"
    detected_language: str = "unknown"
    processing_time: float

# --- 全局实例 ---
whisper_asr = WhisperASR()

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动与关闭管理"""
    try:
        logger.info("🚀 启动服务，正在加载模型...")
        device_info = Config.get_device_info()
        logger.info(f"📊 硬件环境: {device_info}")
        
        # 预加载模型
        whisper_asr.load_models()
        logger.info("✅ 服务启动完成，模型已就绪")
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    finally:
        logger.info("🛑 服务关闭，正在清理显存...")
        whisper_asr.clear_gpu_memory()

# --- FastAPI 应用 ---
app = FastAPI(
    title="Whisper ASR API (LLM Enhanced)",
    description="基于 Whisper Large-v3 + Qwen LLM 的高精度语音识别服务",
    version="2.0.0",
    lifespan=lifespan
)

# --- 核心接口 ---

@app.post("/transcribe", response_model=AudioResponse, summary="语音转文字(统一接口)")
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件 (wav, mp3, m4a, flac, ogg, aac)"),
    forced_language: Optional[str] = Query(None, description="强制指定语言: zh, en, yue, auto"),
    session_id: Optional[str] = Query(None, description="会话ID (用于维持上下文连贯性)"),
    use_context: bool = Query(True, description="是否启用上下文记忆")
):
    """
    统一的音频转录接口。
    - 支持 VAD 语音活动检测
    - 支持 LLM 语义润色
    - 支持 多轮对话上下文记忆
    """
    try:
        if not whisper_asr.model_loaded:
            raise HTTPException(status_code=503, detail="模型正在加载中，请稍后再试")
        
        start_time = time.time()
        
        # 1. 格式验证
        allowed_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')
        if not file.filename.lower().endswith(allowed_exts):
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file.filename}")
        
        # 2. 读取文件
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="文件内容为空")
        
        logger.info(f"📥 接收请求 | 文件: {file.filename} | 大小: {len(audio_bytes)/1024:.1f}KB | 会话: {session_id}")
        
        # 3. 异步执行转录 (避免阻塞主线程)
        loop = asyncio.get_event_loop()
        transcription, detected_lang = await loop.run_in_executor(
            None, 
            whisper_asr.transcribe_audio, 
            audio_bytes, 
            forced_language,
            session_id,
            use_context
        )
        
        process_time = time.time() - start_time
        logger.info(f"📤 处理完成 | 耗时: {process_time:.2f}s | 语言: {detected_lang}")
        
        return AudioResponse(
            transcription=transcription, 
            language="auto",
            detected_language=detected_lang,
            processing_time=process_time
        )
        
    except torch.cuda.OutOfMemoryError:
        logger.critical("🚨 GPU 显存不足，尝试紧急清理")
        whisper_asr.clear_gpu_memory()
        raise HTTPException(status_code=500, detail="服务器显存不足，请稍后重试")
    except Exception as e:
        logger.error(f"❌ 处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 上下文管理接口 ---

@app.delete("/context/{session_id}", summary="清除指定会话记忆")
async def clear_session_context(session_id: str):
    whisper_asr.clear_context_cache(session_id)
    return {"status": "success", "message": f"会话 {session_id} 上下文已清除"}

@app.delete("/context", summary="清除所有会话记忆")
async def clear_all_context():
    whisper_asr.clear_context_cache()
    return {"status": "success", "message": "所有上下文缓存已重置"}

# --- 辅助接口 ---

@app.get("/", summary="服务状态")
async def root():
    return {
        "service": "Whisper ASR Pro",
        "status": "running" if whisper_asr.model_loaded else "loading",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "features": ["VAD", "BeamSearch", "LLM-Correction", "Context-Aware"]
    }

@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "uptime": time.time()}

if __name__ == "__main__":
    logger.info(f"🚀 服务启动中 -> http://{Config.API['host']}:{Config.API['port']}")
    uvicorn.run(
        app,
        host=Config.API["host"],
        port=Config.API["port"],
        timeout_keep_alive=Config.API["timeout_keep_alive"],
        log_level=Config.API["log_level"]
    )