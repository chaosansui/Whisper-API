import logging
import re
from typing import Dict, List, Tuple
from vocabulary import UnifiedVocabulary

logger = logging.getLogger(__name__)

class PostProcessor:
    """后处理器 - 负责文本清理和统一词汇矫正"""
    
    def __init__(self):
        self.vocabulary = UnifiedVocabulary()
        logger.info("✅ 后处理器初始化完成")

    def clean_text(self, text: str, language: str = "auto") -> str:
        """清理和矫正文本 - 不区分语言，统一矫正"""
        if not text or not text.strip():
            return ""
        
        # 记录原始文本用于调试
        original_text = text
        
        # 基础文本清理
        cleaned_text = self._basic_clean(text)
        
        corrected_text = self._apply_unified_corrections(cleaned_text)
        
    
        final_text = self._final_clean(corrected_text)
        
        if original_text != final_text:
            logger.info(f"📝 文本矫正前后对比:")
            logger.info(f"   原始: {original_text}")
            logger.info(f"   矫正: {final_text}")
        
        return final_text

    def _basic_clean(self, text: str) -> str:
        """基础文本清理"""
        if not text:
            return ""
        
        text = ' '.join(text.split())
        
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\-]', '', text)
        
        return text.strip()

    def _apply_unified_corrections(self, text: str) -> str:
        """应用统一词汇矫正 - 简化版本"""
        if not text:
            return ""
        
        # 获取所有矫正词汇
        corrections = self.vocabulary.get_all_corrections()
        
        if not corrections:
            return text
        
        corrected_text = text
        corrections_applied = []
        
        sorted_corrections = sorted(corrections.items())
        
        for wrong, correct in sorted_corrections:
            before_count = corrected_text.count(wrong)
            
            if before_count > 0:
                corrected_text = corrected_text.replace(wrong, correct)
                corrections_applied.append((wrong, correct, before_count))
        
        if corrections_applied:
            total_corrections = sum(count for _, _, count in corrections_applied)
            logger.info(f"🎯 统一词汇矫正: 应用了 {len(corrections_applied)} 种矫正，共 {total_corrections} 处")
            
            for wrong, correct, count in corrections_applied:
                logger.info(f"   '{wrong}' -> '{correct}' (x{count})")
        
        return corrected_text

    def _final_clean(self, text: str) -> str:
        """最终文本清理"""
        if not text:
            return ""
        
        text = text.strip()

        if text and text[-1] not in '.!?。！？':
            text += '.'
        
        return text

    def debug_corrections(self, text: str) -> Dict[str, any]:
        """调试词汇矫正情况"""
        corrections = self.vocabulary.get_all_corrections()
        debug_info = {
            "original_text": text,
            "total_correction_rules": len(corrections),
            "matched_corrections": [],
            "corrected_text": self.clean_text(text)
        }
        
        for wrong, correct in corrections.items():
            if wrong in text:
                debug_info["matched_corrections"].append({
                    "wrong": wrong,
                    "correct": correct,
                    "count": text.count(wrong)
                })
        
        return debug_info

    def list_all_corrections(self) -> List[Tuple[str, str]]:
        """列出所有矫正规则"""
        corrections = self.vocabulary.get_all_corrections()
        return [(k, v) for k, v in corrections.items()]

    def add_custom_correction(self, wrong: str, correct: str):
        """添加自定义词汇矫正"""
        self.vocabulary.add_custom_correction(wrong, correct)
        logger.info(f"✅ 已添加矫正规则: '{wrong}' -> '{correct}'")

    def batch_add_corrections(self, corrections: Dict[str, str]):
        """批量添加词汇矫正"""
        self.vocabulary.batch_add_corrections(corrections)
        logger.info(f"✅ 批量添加 {len(corrections)} 个矫正规则")

    def remove_correction(self, word: str):
        """移除词汇矫正"""
        self.vocabulary.remove_correction(word)
        logger.info(f"✅ 已移除矫正规则: '{word}'")

    def get_vocabulary_stats(self) -> Dict[str, int]:
        """获取词汇表统计"""
        stats = self.vocabulary.get_vocabulary_stats()
        logger.info(f"📊 词汇表统计: {stats}")
        return stats

    def search_corrections(self, keyword: str) -> Dict[str, str]:
        """搜索相关矫正项"""
        results = self.vocabulary.search_corrections(keyword)
        logger.info(f"🔍 搜索 '{keyword}': 找到 {len(results)} 个相关矫正")
        return results