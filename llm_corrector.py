import logging
import requests
import time
from config import Config

logger = logging.getLogger(__name__)

class LLMCorrector:
    def __init__(self, vocabulary=None):

        self.api_url = Config.Local_model_url
        self.model_name = Config.Local_model_name
        self.vocab_str = "、".join(vocabulary[:50]) if vocabulary else ""

    def build_prompt(self, raw_text):
        """构建 Prompt (无词汇表纯模型模式)"""
        system_prompt = f"""你是一个智能语音转写文本润色助手。
任务：接收一段语音识别生成的原始文本，修正其中的同音错别字、标点错误和不通顺的地方，这是粤语关于金融借贷领域，可能包含粤语表达，注意逻辑顺序，请根据上下文进行合理修正。

【处理原则】：
1. **尊重原意**：绝对不要删减信息，不要随意扩写，保持原句的语气。
2. **专业性修正**：根据上下文，提到类似"分KeyFinance"改为"FunkiFinance"和"www.funky.com.jk"改为"www.funki.com.hk"等领域特定的同音错误。
3. **格式规范**：修正标点符号，统一中英文混排格式（如在中文和英文单词间增加空格）。
4. **直接输出**：只输出修正后的最终文本，不要包含任何解释或开场白。

原始文本：
{raw_text}
"""
        return system_prompt

    def correct(self, text):
        """调用 vLLM 进行纠错"""
        # 文本太短没必要修，比如"嗯"、"啊"
        if not text or len(text) < 2:
            return text

        start_time = time.time()
        
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.build_prompt(text)},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1, # 低温保证严谨
                "max_tokens": 1024,
                "stream": False
            }

            # 发送请求
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=100 
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result['choices'][0]['message']['content'].strip()
                logger.info(f"🤖 LLM修正 ({time.time()-start_time:.2f}s): {text[:15]}... -> {corrected_text[:15]}...")
                return corrected_text
            else:
                logger.warning(f"LLM API返回错误: {response.status_code}")
                return text

        except Exception as e:
            logger.warning(f"LLM 调用失败 (跳过修正): {e}")
            return text