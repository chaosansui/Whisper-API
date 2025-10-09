import re
import logging
from typing import Optional
from vocabulary import Vocabulary

logger = logging.getLogger(__name__)

class PostProcessor:
    """后处理器"""
    
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.setup_patterns()
    
    def setup_patterns(self):
        """设置正则表达式模式"""
        # 清理模式
        self.clean_patterns = [
            (r'[^\w\s\u4e00-\u9fff.,!?;:，。！？；：\-()（）]', ''),  # 移除非法字符
            (r'\s+', ' '),  # 合并多个空格
            (r'([.!?。！？])([^\s])', r'\1 \2'),  # 标点后添加空格
        ]
        
        # 粤语特定模式
        self.cantonese_patterns = [
            (r'(唔)([^係要好錯])', r'\1 \2'),  # 粤语"唔"后加空格
            (r'(嘅)([^\.。!！?？,，])', r'\1 \2'),  # 粤语"嘅"后加空格
        ]
        
        # 英文特定模式
        self.english_patterns = [
            (r'\b(\w+)\s+\1\b', r'\1'),  # 移除重复单词
            (r'\bi\b', 'I'),  # 小写i改为大写
        ]

    def clean_text(self, text: str, language: Optional[str] = None) -> str:
        """清理转录文本"""
        if not text or not isinstance(text, str):
            return ""
        
        # 基础清理
        for pattern, replacement in self.clean_patterns:
            text = re.sub(pattern, replacement, text)
        
        text = text.strip()
        
        # 语言特定处理
        if language == "yue":
            text = self.postprocess_cantonese(text)
        elif language == "en":
            text = self.postprocess_english(text)
        elif language == "zh":
            text = self.postprocess_chinese(text)
        
        # 首字母大写
        if text and language in ["en"]:
            text = text[0].upper() + text[1:]
        
        return text

    def postprocess_cantonese(self, text: str) -> str:
        """粤语转录后处理"""
        if not text:
            return text
        
        # 词汇替换
        corrections = self.vocabulary.get_corrections_for_language("yue")
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # 模式处理
        for pattern, replacement in self.cantonese_patterns:
            text = re.sub(pattern, replacement, text)
        
        logger.debug(f"粤语后处理结果: {text}")
        return text

    def postprocess_english(self, text: str) -> str:
        """英文转录后处理"""
        if not text:
            return text
        
        # 词汇替换
        corrections = self.vocabulary.get_corrections_for_language("en")
        for wrong, correct in corrections.items():
            # 使用单词边界确保完整单词匹配
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text)
        
        # 模式处理
        for pattern, replacement in self.english_patterns:
            text = re.sub(pattern, replacement, text)
        
        logger.debug(f"英文后处理结果: {text}")
        return text

    def postprocess_chinese(self, text: str) -> str:
        """中文转录后处理"""
        if not text:
            return text
        
        # 词汇替换
        corrections = self.vocabulary.get_corrections_for_language("zh")
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        logger.debug(f"中文后处理结果: {text}")
        return text

    def add_custom_correction(self, language: str, wrong: str, correct: str):
        """添加自定义修正"""
        self.vocabulary.add_custom_vocabulary(language, {wrong: correct})
        logger.info(f"添加自定义修正: {language} - {wrong} -> {correct}")

    def batch_add_corrections(self, language: str, corrections: dict):
        """批量添加修正"""
        self.vocabulary.add_custom_vocabulary(language, corrections)
        logger.info(f"批量添加 {len(corrections)} 个 {language} 修正")