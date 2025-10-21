import logging
from typing import Dict, Optional
from vocabulary import Vocabulary

logger = logging.getLogger(__name__)

class PostProcessor:
    """后处理器 - 负责文本清理和统一词汇矫正"""
    
    def __init__(self):
        self.vocabulary = Vocabulary()
        logger.info("✅ 后处理器初始化完成")

    def clean_text(self, text: str, language: str = "auto") -> str:
        """清理和矫正文本 - 不区分语言，统一矫正"""
        if not text or not text.strip():
            return ""
        
        # 基础文本清理
        cleaned_text = self._basic_clean(text)
        
        # 🎯 统一词汇矫正（不区分语言）
        corrected_text = self._apply_unified_corrections(cleaned_text)
        
        # 最终格式清理
        final_text = self._final_clean(corrected_text)
        
        return final_text

    def _basic_clean(self, text: str) -> str:
        """基础文本清理"""
        if not text:
            return ""
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        # 移除特殊字符但保留标点
        import re
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\-]', '', text)
        
        return text.strip()

    def _apply_unified_corrections(self, text: str) -> str:
        """应用统一词汇矫正 - 修复包含空格的短语匹配"""
        if not text:
            return ""
        
        # 获取所有矫正词汇
        corrections = self.vocabulary.get_all_corrections()
        
        if not corrections:
            return text
        
        # 按短语长度降序排序（先匹配长短语）
        sorted_corrections = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
        
        corrected_text = text
        corrections_applied = []
        
        for wrong, correct in sorted_corrections:
            # 🎯 修复：对包含空格的短语使用直接字符串替换
            if ' ' in wrong:
                # 对于包含空格的短语，使用直接替换
                if wrong in corrected_text:
                    corrected_text = corrected_text.replace(wrong, correct)
                    corrections_applied.append((wrong, correct))
            else:
                # 对于单个单词，使用单词边界确保准确替换
                import re
                pattern = r'\b' + re.escape(wrong) + r'\b'
                if re.search(pattern, corrected_text):
                    corrected_text = re.sub(pattern, correct, corrected_text)
                    corrections_applied.append((wrong, correct))
        
        # 记录矫正情况（如果有变化）
        if corrections_applied:
            logger.info(f"🎯 统一词汇矫正: 应用了 {len(corrections_applied)} 个矫正")
            for wrong, correct in corrections_applied:
                logger.info(f"   '{wrong}' -> '{correct}'")
        
        return corrected_text

    def _final_clean(self, text: str) -> str:
        """最终文本清理"""
        if not text:
            return ""
        
        # 移除首尾空格
        text = text.strip()
        
        # 确保句子以标点结尾
        if text and text[-1] not in '.!?。！？':
            text += '.'
        
        return text

    def add_custom_correction(self, wrong: str, correct: str):
        """添加自定义词汇矫正"""
        self.vocabulary.add_custom_correction(wrong, correct)

    def batch_add_corrections(self, corrections: Dict[str, str]):
        """批量添加词汇矫正"""
        self.vocabulary.batch_add_corrections(corrections)

    def remove_correction(self, word: str):
        """移除词汇矫正"""
        self.vocabulary.remove_correction(word)

    def get_vocabulary_stats(self) -> Dict[str, int]:
        """获取词汇表统计"""
        return self.vocabulary.get_vocabulary_stats()

    def search_corrections(self, keyword: str) -> Dict[str, str]:
        """搜索相关矫正项"""
        return self.vocabulary.search_corrections(keyword)