import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class UnifiedVocabulary:
    
    def __init__(self):
        self.corrections = self._load_unified_vocabulary()
        logger.info(f"✅ 统一词汇表加载完成: 共 {len(self.corrections)} 个矫正项")

    def _load_unified_vocabulary(self) -> Dict[str, str]:
        """加载统一词汇表"""
        corrections = {
            'Punky Finance': 'Funky Finance',
            "www. funky. com. jk": "www.funky.com.hk",
            '新聞證': '身份證',
            '銀碼': '金額',
            'OK': '額度',
            '票': '支票',
            '刷成': '調整為',
            '改資料': '修改資料',
            '私人貸款': '私人貸款',
            '預算': '預算',
            '現金': '現金',
            '退了': '推遲',
            '收': '收取',
            
            "是不是": "係唔係", "这样": "咁樣", "那个": "嗰個", 
            "这里": "呢度", "为什么": "點解", "没有": "冇",
            "的": "嘅", "他": "佢", "我们": "我哋", "什么": "咩",
            
            "there": "their", "your": "you're", "its": "it's", 
            "to": "too", "then": "than", "weather": "whether",
            "mortage": "mortgage", "intrest": "interest",
            
            "登陆": "登录", "帐号": "账号", "其它": "其他",
            "部份": "部分", "重起": "重启", "按装": "安装",
            
            "gonna": "going to", "wanna": "want to", 
            "kinda": "kind of", "sorta": "sort of",
            
            "账户": "戶口", "密码": "密碼", "转账": "過數", 
            "余额": "餘額", "贷款": "貸款", "投资": "投資",
        }
        
        return corrections

    def correct_text(self, text: str) -> str:
        """对文本应用词汇矫正"""
        original_text = text
        corrected_text = text
        
        applied_corrections = []
        for wrong, correct in self.corrections.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                applied_corrections.append(f"'{wrong}' -> '{correct}'")
        
        if applied_corrections:
            logger.info(f"🎯 统一词汇矫正: 应用了 {len(applied_corrections)} 个矫正")
            for correction in applied_corrections:
                logger.info(f"   {correction}")
            logger.info(f"📝 矫正前: {original_text}")
            logger.info(f"📝 矫正后: {corrected_text}")
        
        return corrected_text

    def get_all_corrections(self) -> Dict[str, str]:
        return self.corrections

    def add_custom_correction(self, wrong: str, correct: str):
        self.corrections[wrong] = correct

    def batch_add_corrections(self, corrections_dict: Dict[str, str]):
        self.corrections.update(corrections_dict)

    def remove_correction(self, word: str):
        if word in self.corrections:
            del self.corrections[word]

    def get_vocabulary_stats(self) -> Dict[str, int]:
        return {"total_corrections": len(self.corrections)}

    def search_corrections(self, keyword: str) -> Dict[str, str]:
        results = {}
        for wrong, correct in self.corrections.items():
            if keyword.lower() in wrong.lower() or keyword.lower() in correct.lower():
                results[wrong] = correct
        return results