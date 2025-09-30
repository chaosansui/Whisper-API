import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import time
import traceback
from typing import Optional, List, Dict
from asr_service import WhisperASR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

class AudioResponse(BaseModel):
    transcriptions: List[Dict[str, str | float]]
    detected_language: str = "unknown"

whisperAsr = None

@asynccontextmanager
async def lifespan(app):

    global whisperAsr
    try:
        logger.info("🚀 初始化 WhisperASR...")
        whisperAsr = WhisperASR()
        logger.info("🚀 启动服务，加载模型中...")
        whisperAsr.load_models() 
        logger.info("✅ 服务启动完成")
        yield
    except Exception as e:
        logger.error(f"❌ WhisperASR 初始化或模型加载失败: {str(e)}")
        logger.error(f"详细堆栈: {traceback.format_exc()}")
        raise
    finally:
        # 关闭时清理资源
        if whisperAsr:
            logger.INFO("🛑 服务关闭，清理资源...")
            whisperAsr.clear_gpu_memory()

# 创建 FastAPI 应用（只创建一次）
app = FastAPI(
    title="Whisper + 语音识别 API",
    description="基于 Whisper 和说话人分割的语音转录服务，优化客服场景（粤语/英语）",
    version="2.7.2",
    lifespan=lifespan  # 正确设置 lifespan
)

@app.post("/transcribe", response_model=AudioResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    forced_language: Optional[str] = Query(None, description="强制指定语言，如：cantonese, english")
):
    try:
        if not whisperAsr or not whisperAsr.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")
        
        logger.info(f"收到音频: {file.filename}, 大小: {len(audio_bytes)} 字节")
        loop = asyncio.get_event_loop()
        transcriptions, detected_language = await loop.run_in_executor(
            whisperAsr.executor, whisperAsr.process_call_recording, audio_bytes, forced_language
        )
        total_time = time.time() - start_time
        logger.info(f"请求处理完成，耗时: {total_time:.2f}秒")
        
        return AudioResponse(
            transcriptions=transcriptions,
            detected_language=detected_language
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        logger.error(f"详细堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/")
async def root():
    status = "运行中" if whisperAsr and whisperAsr.model_loaded else "启动中"
    return {
        "message": "Whisper + pyannote 语音识别服务 - 客服优化版",
        "status": status,
        "supported_languages": ["auto", "cantonese", "english", "chinese"],
        "endpoints": {
            "transcribe": "POST /transcribe - 上传音频进行转录和说话人分割",
            "health": "GET /health - 服务健康检查",
            "INFO": "GET /INFO - 模型信息",
            "model_status": "GET /model_status - 模型状态检查"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if whisperAsr and whisperAsr.model_loaded else "unhealthy",
        "model_loaded": whisperAsr.model_loaded if whisperAsr else False,
        "gpu_available": whisperAsr.torch.cuda.is_available() if whisperAsr else False,
        "timestamp": time.time()
    }

@app.get("/INFO")
async def model_INFO():
    if not whisperAsr or not whisperAsr.model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    diarization_status = "已加载" if whisperAsr.diarization_pipeline is not None else "未加载"
    
    return {
        "model_name": "Whisper-large-v3-cantonese + pyannote/segmentation-3.0",
        "language": "粤语/英语，优化客服场景",
        "special_optimization": "说话人分割，粤语/英语混合，低延迟分割",
        "device": str(whisperAsr.device),
        "diarization_model": diarization_status,
        "model_loaded": True,
        "optimizations": ["n_mels=256", "beam_size=10-12", "temperature=0.15", "pyannote 分割", "双说话人优化"]
    }

@app.get("/model_status")
async def model_status():
    """检查模型状态"""
    if not whisperAsr:
        return {
            "status": "服务未初始化",
            "whisper_model": "未加载",
            "diarization_model": "未加载",
            "local_model_path": "未知"
        }
    
    diarization_status = "已加载" if whisperAsr.diarization_pipeline is not None else "未加载"
    local_path = getattr(whisperAsr, 'local_segmentation_path', '未设置')
    
    return {
        "status": "运行中" if whisperAsr.model_loaded else "启动中",
        "whisper_model": "已加载" if whisperAsr.whisper_model is not None else "未加载",
        "diarization_model": diarization_status,
        "local_model_path": local_path,
        "model_loaded": whisperAsr.model_loaded
    }

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8008,
            timeout_keep_alive=300,
        )
    except Exception as e:
        logger.error(f"❌ Uvicorn 启动失败: {str(e)}")
        logger.error(f"详细堆栈: {traceback.format_exc()}")