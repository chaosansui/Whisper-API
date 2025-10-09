import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
import asyncio
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import time
from whisper_asr import WhisperASR
from config import API_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 定义响应模型
class AudioResponse(BaseModel):
    transcription: str
    language: str = "auto"
    detected_language: str = "unknown"

class CorrectionRequest(BaseModel):
    language: str
    wrong: str
    correct: str

class BatchCorrectionRequest(BaseModel):
    language: str
    corrections: dict

# 创建全局实例
whisperAsr = WhisperASR()

@asynccontextmanager
async def lifespan(app):
    """FastAPI 生命周期管理"""
    try:
        logger.info("🚀 启动服务，加载模型中...")
        whisperAsr.load_models()
        logger.info("✅ 服务启动完成")
        yield
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    finally:
        logger.info("🛑 服务关闭，清理资源...")
        whisperAsr.clear_gpu_memory()

# 创建 FastAPI 应用
app = FastAPI(
    title="Whisper语音识别API",
    description="基于Whisper的多语言语音识别服务，优化智能客服场景（粤语+英文，含口音和噪音）",
    version="2.7.1",
    lifespan=lifespan
)

@app.post("/transcribe", response_model=AudioResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    forced_language: Optional[str] = Query(None, description="强制指定语言，如：cantonese, english")
):
    """音频转录接口，支持强制语言设置"""
    try:
        if not whisperAsr.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")
        
        start_time = time.time()
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        
        logger.info(f"收到音频文件: {file.filename}, 大小: {len(audio_bytes)} bytes")
        
        loop = asyncio.get_event_loop()
        transcription, detected_language = await loop.run_in_executor(
            whisperAsr.executor, whisperAsr.transcribe_audio, audio_bytes, forced_language
        )
        
        total_time = time.time() - start_time
        logger.info(f"请求处理完成，总耗时: {total_time:.2f} 秒")
        
        return AudioResponse(
            transcription=transcription, 
            language="auto",
            detected_language=detected_language
        )
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        whisperAsr.clear_gpu_memory()
        raise HTTPException(status_code=500, detail="GPU内存不足")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/vocabulary/add")
async def add_correction(correction: CorrectionRequest):
    """添加自定义词汇修正"""
    try:
        whisperAsr.add_custom_correction(
            correction.language,
            correction.wrong,
            correction.correct
        )
        return {"message": "修正添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加修正失败: {str(e)}")

@app.post("/vocabulary/batch_add")
async def batch_add_corrections(batch_correction: BatchCorrectionRequest):
    """批量添加词汇修正"""
    try:
        whisperAsr.batch_add_corrections(
            batch_correction.language,
            batch_correction.corrections
        )
        return {"message": f"批量添加 {len(batch_correction.corrections)} 个修正成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量添加修正失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Whisper多语言语音识别服务 - 智能客服优化版",
        "status": "运行中" if whisperAsr.model_loaded else "启动中",
        "supported_languages": ["auto", "cantonese", "english", "chinese", "japanese", "korean", "vietnamese"],
        "endpoints": {
            "transcribe": "POST /transcribe - 上传音频进行转录",
            "health": "GET /health - 服务健康检查",
            "info": "GET /info - 模型信息",
            "vocabulary_add": "POST /vocabulary/add - 添加词汇修正",
            "vocabulary_batch_add": "POST /vocabulary/batch_add - 批量添加词汇修正"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if whisperAsr.model_loaded else "unhealthy",
        "model_loaded": whisperAsr.model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }

@app.get("/info")
async def model_info():
    """模型信息接口"""
    if not whisperAsr.model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {
        "model_name": "Whisper-large-v3",
        "language": "多语言自动检测，优化粤语和英文（智能客服场景）",
        "special_optimization": "粤语+英文混合，口音和噪音处理，长音频分块，无语言标签输出",
        "device": str(whisperAsr.device),
        "model_loaded": True,
        "optimizations": ["n_mels=256", "beam_size=10", "temperature=0.2", "动态降噪", "自适应VAD", "粤语优化", "英文优化", "混合语言处理", "长音频分块", "无语言标签"]
    }

if __name__ == "__main__":
    logger.info("Starting Whisper ASR Server with Cantonese+English optimization for customer service...")
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        timeout_keep_alive=API_CONFIG["timeout_keep_alive"],
        log_level=API_CONFIG["log_level"]
    )