import os
# 设置 CUDA 设备和内存管理策略
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import time
from whisper_asr import WhisperASR
from config import Config

# 配置日志
logging.basicConfig(level=Config.LOGGING["level"], format=Config.LOGGING["format"])
logger = logging.getLogger(__name__)

# 定义响应模型
class AudioResponse(BaseModel):
    transcription: str
    language: str = "auto"
    detected_language: str = "unknown"

class CorrectionRequest(BaseModel):
    wrong: str
    correct: str

class BatchCorrectionRequest(BaseModel):
    corrections: dict

# 创建全局实例
whisper_asr = WhisperASR()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理"""
    try:
        logger.info("🚀 启动服务，加载模型中...")
        device_info = Config.get_device_info()
        logger.info(f"📊 设备信息: {device_info}")
        
        whisper_asr.load_models()
        logger.info("✅ 服务启动完成")
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    finally:
        logger.info("🛑 服务关闭，清理资源...")
        whisper_asr.clear_gpu_memory()

# 创建 FastAPI 应用
app = FastAPI(
    title="Whisper语音识别API",
    description="基于Whisper的多语言语音识别服务",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/transcribe", response_model=AudioResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件 (支持 wav, mp3, m4a, flac, ogg, aac)"),
    forced_language: Optional[str] = Query(None, description="强制指定语言: yue(粤语), zh(中文), en(英文), auto(自动检测)")
):
    """音频转录接口"""
    try:
        if not whisper_asr.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")
        
        start_time = time.time()
        
        # 文件格式验证
        allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail=f"不支持的文件格式")
        
        # 文件大小验证
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        
        logger.info(f"收到音频文件: {file.filename}, 大小: {len(audio_bytes)} bytes")
        
        # 异步处理转录
        loop = asyncio.get_event_loop()
        transcription, detected_language = await loop.run_in_executor(
            None, 
            whisper_asr.transcribe_audio, 
            audio_bytes, 
            forced_language
        )
        
        total_time = time.time() - start_time
        logger.info(f"请求处理完成 - 耗时: {total_time:.2f}秒, 语言: {detected_language}")
        
        return AudioResponse(
            transcription=transcription, 
            language="auto",
            detected_language=detected_language
        )
        
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU内存不足，清理内存")
        whisper_asr.clear_gpu_memory()
        raise HTTPException(status_code=500, detail="GPU内存不足，请重试")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/vocabulary/add", summary="添加词汇修正")
async def add_correction(correction: CorrectionRequest):
    """添加自定义词汇修正"""
    try:
        whisper_asr.add_custom_correction(correction.wrong, correction.correct)
        
        logger.info(f"添加词汇修正: '{correction.wrong}' -> '{correction.correct}'")
        return {
            "message": "修正添加成功",
            "correction": f"'{correction.wrong}' -> '{correction.correct}'"
        }
        
    except Exception as e:
        logger.error(f"添加修正失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加修正失败: {str(e)}")

@app.post("/vocabulary/batch_add", summary="批量添加词汇修正")
async def batch_add_corrections(batch_correction: BatchCorrectionRequest):
    """批量添加词汇修正"""
    try:
        if not isinstance(batch_correction.corrections, dict):
            raise HTTPException(status_code=400, detail="修正数据格式错误，应为字典类型")
        
        whisper_asr.batch_add_corrections(batch_correction.corrections)
        
        logger.info(f"批量添加 {len(batch_correction.corrections)} 个修正")
        return {
            "message": f"批量添加 {len(batch_correction.corrections)} 个修正成功",
            "count": len(batch_correction.corrections)
        }
        
    except Exception as e:
        logger.error(f"批量添加修正失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量添加修正失败: {str(e)}")

@app.get("/vocabulary/stats", summary="获取词汇表统计")
async def get_vocabulary_stats():
    """获取词汇表统计信息"""
    try:
        stats = whisper_asr.get_vocabulary_stats()
        return {
            "message": "词汇表统计获取成功",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"获取词汇表统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

@app.get("/", summary="服务首页")
async def root():
    """根路径 - 服务信息"""
    device_info = Config.get_device_info()
    
    return {
        "message": "Whisper多语言语音识别服务",
        "version": "1.0.0",
        "status": "运行中" if whisper_asr.model_loaded else "启动中",
        "device_info": device_info,
        "supported_languages": ["auto", "yue", "zh", "en"],
        "features": [
            "多语言语音识别",
            "自动语言检测", 
            "长音频分块处理",
            "统一词汇矫正",
            "内存使用优化"
        ]
    }

@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if whisper_asr.model_loaded else "unhealthy",
        "timestamp": time.time(),
        "model_loaded": whisper_asr.model_loaded
    }

@app.get("/info", summary="模型信息")
async def model_info():
    """模型信息接口"""
    if not whisper_asr.model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return whisper_asr.get_model_info()
# 在现有的路由后面添加以下新路由

@app.post("/transcribe_with_context", response_model=AudioResponse)
async def transcribe_audio_with_context(
    file: UploadFile = File(..., description="音频文件"),
    forced_language: Optional[str] = Query(None, description="强制指定语言"),
    session_id: Optional[str] = Query(None, description="会话ID，用于维持上下文"),
    use_context: bool = Query(True, description="是否使用上下文提示词")
):
    """支持上下文连贯性的音频转录接口"""
    try:
        if not whisper_asr.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")
        
        start_time = time.time()
        
        # 文件格式验证
        allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail=f"不支持的文件格式")
        
        # 文件大小验证
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        
        logger.info(f"收到音频文件: {file.filename}, 大小: {len(audio_bytes)} bytes, 会话ID: {session_id}")
        
        # 异步处理转录
        loop = asyncio.get_event_loop()
        transcription, detected_language = await loop.run_in_executor(
            None, 
            whisper_asr.transcribe_audio, 
            audio_bytes, 
            forced_language,
            session_id,
            use_context
        )
        
        total_time = time.time() - start_time
        logger.info(f"上下文转录完成 - 耗时: {total_time:.2f}秒, 语言: {detected_language}")
        
        return AudioResponse(
            transcription=transcription, 
            language="auto",
            detected_language=detected_language
        )
        
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU内存不足，清理内存")
        whisper_asr.clear_gpu_memory()
        raise HTTPException(status_code=500, detail="GPU内存不足，请重试")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.delete("/context/{session_id}", summary="清理指定会话的上下文")
async def clear_session_context(session_id: str):
    """清理指定会话的上下文缓存"""
    try:
        whisper_asr.clear_context_cache(session_id)
        return {"message": f"已清理会话 {session_id} 的上下文缓存"}
    except Exception as e:
        logger.error(f"清理上下文缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

@app.delete("/context", summary="清理所有上下文缓存")
async def clear_all_context():
    """清理所有上下文缓存"""
    try:
        whisper_asr.clear_context_cache()
        return {"message": "已清理所有上下文缓存"}
    except Exception as e:
        logger.error(f"清理上下文缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

@app.get("/context/stats", summary="获取上下文缓存统计")
async def get_context_stats():
    """获取上下文缓存统计信息"""
    try:
        model_info = whisper_asr.get_model_info()
        return {
            "context_cache_size": model_info.get("context_cache_size", 0),
            "active_sessions": list(whisper_asr.context_cache.keys()) if hasattr(whisper_asr, 'context_cache') else []
        }
    except Exception as e:
        logger.error(f"获取上下文统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting Whisper ASR Server...")
    
    uvicorn.run(
        app,
        host=Config.API["host"],
        port=Config.API["port"],
        timeout_keep_alive=Config.API["timeout_keep_alive"],
        log_level=Config.API["log_level"]
    )