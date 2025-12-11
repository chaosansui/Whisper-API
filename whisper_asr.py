import torch
import numpy as np
import time
import logging
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import Config
from audio_processor import AudioProcessor
from llm_corrector import LLMCorrector

logger = logging.getLogger(__name__)

class WhisperASR:
    def __init__(self):
        self.device = Config.DEVICE
        self.whisper_model = None
        self.processor = None
        self.audio_processor = AudioProcessor()
        self.model_loaded = False
        self.context_cache = {}
        self.llm_corrector = LLMCorrector(vocabulary=[])
        
    def load_models(self):
        if self.model_loaded: return
            
        try:
            logger.info("🔄 加载 Whisper 模型...")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.processor = WhisperProcessor.from_pretrained(Config.MODEL_PATH)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                Config.MODEL_PATH,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            ).to(self.device)
            self.whisper_model.eval()
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            self.model_loaded = True
            logger.info("✅ 模型加载完成")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            raise

    def transcribe_audio(self, audio_bytes: bytes, forced_language: str = None, 
                    session_id: str = None, use_context: bool = True) -> tuple:
        if not self.model_loaded: return "模型未加载", "error"
            
        try:
            start_time = time.time()
            
            # 1. 预处理
            audio_np = self.audio_processor.preprocess_audio(audio_bytes)
            
            # 2. 验证
            if not self.audio_processor.validate_audio(audio_np):
                return "未检测到有效语音", "auto"
            
            # 3. 上下文
            prompt_text = ""
            if use_context and session_id and session_id in self.context_cache:
                prompt_text = self.context_cache[session_id]
            
            # 4. VAD
            speech_segments = self.audio_processor.vad_segmentation(audio_np)
            if not speech_segments:
                merged_segments = [self._pad_to_min_duration(audio_np, 15.0)]
            else:
                merged_segments = self._merge_short_segments(speech_segments, min_duration=15.0)
                if not merged_segments:
                    merged_segments = [self._pad_to_min_duration(audio_np, 15.0)]
            
            # 5. Whisper 转录
            if len(merged_segments) == 1:
                transcription, detected_language = self._transcribe_chunk(
                    merged_segments[0], forced_language, is_first_chunk=True, prompt_text=prompt_text
                )
            else:
                transcription, detected_language = self._transcribe_vad_segments(
                    merged_segments, forced_language, prompt_text
                )
            
            # 6. LLM 润色
            if transcription and transcription.strip() and detected_language in ["zh", "yue", "auto"]:
                if detected_language == "zh":
                    transcription = transcription.replace(" ", "")
                
                logger.info("🚀 调用 vLLM 润色...")
                corrected_text = self.llm_corrector.correct(transcription)
                if corrected_text:
                    transcription = corrected_text

            # 7. 更新上下文
            if use_context and session_id and transcription:
                self._update_context_cache(session_id, transcription)
            
            total_time = time.time() - start_time
            if transcription:
                logger.info(f"✅ 完成 ({total_time:.2f}s): {transcription[:50]}...")
            
            return transcription, detected_language
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"转录失败: {str(e)}", "error"

    def _merge_short_segments(self, segments: list, min_duration: float = 15.0) -> list:
        if not segments: return []
        if len(segments) == 1:
            return [self._pad_to_min_duration(segments[0], min_duration)] if len(segments[0]) < min_duration*16000 else segments
        
        merged = []
        curr = [segments[0]]
        curr_dur = len(segments[0])/16000
        TARGET = 28.0

        for i in range(1, len(segments)):
            seg = segments[i]
            dur = len(seg)/16000
            if curr_dur + dur <= TARGET:
                curr.append(seg)
                curr_dur += dur
            else:
                merged.append(self._merge_batch_segments(curr))
                curr = [seg]
                curr_dur = dur
        if curr: merged.append(self._merge_batch_segments(curr))
        
        return [self._pad_to_min_duration(s, min_duration) if len(s) < min_duration*16000 else s for s in merged]

    def _merge_batch_segments(self, batch: list) -> np.ndarray:
        if not batch: return np.array([])
        if len(batch) == 1: return batch[0]
        res = []

        gap_noise = np.random.normal(0, 0.0001, int(0.15*16000)).astype(np.float32)
        
        for i, s in enumerate(batch):
            res.append(s)
            if i < len(batch)-1: res.append(gap_noise)
        return np.concatenate(res)

    def _pad_to_min_duration(self, audio: np.ndarray, min_duration: float) -> np.ndarray:
        target = int(min_duration * 16000)
        curr_len = len(audio)
        if curr_len >= target: return audio
        
        padding_len = target - curr_len
        noise = np.random.normal(0, 0.0001, padding_len).astype(np.float32)
        
        return np.concatenate([audio, noise])

    def _transcribe_vad_segments(self, segments, forced_language, initial_prompt):
        all_text = []
        lang = "auto"
        prompt = initial_prompt
        for i, seg in enumerate(segments):
            curr_prompt = prompt[-200:] if len(prompt)>200 else prompt
            txt, l = self._transcribe_chunk(seg, forced_language, i==0, curr_prompt)
            if txt:
                all_text.append(txt)
                prompt += " " + txt
            if i==0: lang = l
        return " ".join(all_text), lang

    def _transcribe_chunk(self, chunk, forced_language, is_first_chunk, prompt_text):
        try:
            
            target = int(30.0 * 16000)
            if len(chunk) < target: 
                padding_len = target - len(chunk)
                noise = np.random.normal(0, 0.0001, padding_len).astype(np.float32)
                chunk = np.concatenate([chunk, noise])
            elif len(chunk) > target: 
                chunk = chunk[:target]
            
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            feat = inputs.input_features.to(self.device)
            mask = inputs.attention_mask.to(self.device)
            if self.device.type=="cuda": feat = feat.half()
            
            gen_kwargs = {
                "max_new_tokens": 400, 
                "num_beams": 5, 
                "repetition_penalty": 1.1, 
                "attention_mask": mask
            }
            if forced_language and forced_language!="auto":
                gen_kwargs["language"] = forced_language
                gen_kwargs["task"] = "transcribe"
                
            with torch.inference_mode():
                ids = self.whisper_model.generate(feat, **gen_kwargs)
            
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            
            if self._is_mostly_chinese(text):
                text = text.replace(" ", "")
                
            detected = "auto"
            if is_first_chunk and not forced_language:
                detected = self._detect_language(ids)
            return text, detected
        except Exception as e:
            logger.error(f"Chunk error: {e}")
            return "", "auto"

    def _is_mostly_chinese(self, text):
        zh_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
        return zh_count > len(text) * 0.5

    def _update_context_cache(self, session_id, text):
        if len(self.context_cache) >= 100:
             self.context_cache.pop(next(iter(self.context_cache)))
        self.context_cache[session_id] = text[-800:]

    def _detect_language(self, ids):
        try:
            tokens = ids[0, :5]
            decoded = self.processor.decode(tokens)
            if "zh" in decoded: return "zh"
            if "en" in decoded: return "en"
            if "yue" in decoded: return "yue"
            return "auto"
        except: return "auto"
    
    def clear_gpu_memory(self):
        if self.device.type == "cuda": torch.cuda.empty_cache()