import logging
from typing import Dict

logger = logging.getLogger(__name__)

class Vocabulary:
    
    def __init__(self):
        self.unified_corrections = self._load_unified_vocabulary()
        logger.info("✅ 统一词汇表加载完成")

    def _load_unified_vocabulary(self) -> Dict[str, str]:
        """加载统一词汇表 - 不区分语言"""
        corrections = {}
        
        # 粤语词汇
        cantonese = {
            "是不是": "係唔係", "这样": "咁樣", "那个": "嗰個", "这里": "呢度",
            "为什么": "點解", "没有": "冇", "的": "嘅", "他": "佢", 
            "我们": "我哋", "什么": "咩", "怎么": "點樣", "现在": "而家",
            "账户": "戶口", "密码": "密碼", "转账": "過數", "余额": "餘額",
            "贷款": "貸款", "信用卡": "信用卡", "投资": "投資", "理财": "理財",
            "登录": "登入", "注册": "登記", "问题": "問題", "解决": "解決",
            "新聞證": " 身份證", "服务": "服務", "设置": "設定", "订单": "訂單"
        }
        
        # 英语词汇
        english = {
            "there": "their", "your": "you're", "its": "it's", 
            "to": "too", "then": "than", "weather": "whether",
            "mortage": "mortgage", "intrest": "interest", 
            "withdrawl": "withdrawal", "deposite": "deposit",
            "Punky Finance": "Funky Finance", "www. funky. com. jk": "www.funky.com.hk",
            "gonna": "going to", "wanna": "want to", "gotta": "got to",
            "kinda": "kind of", "sorta": "sort of"
        }
        
        # 中文词汇
        chinese = {
            "登陆": "登录", "帐号": "账号", "其它": "其他",
            "部份": "部分", "重起": "重启", "按装": "安装"
        }
        
        # 合并所有词汇表
        corrections.update(cantonese)
        corrections.update(english)
        corrections.update(chinese)
        
        logger.info(f"📚 统一词汇表加载完成: 共 {len(corrections)} 个矫正项")
        return corrections

    def get_all_corrections(self) -> Dict[str, str]:
        """获取所有词汇替换表"""
        return self.unified_corrections

    def add_custom_correction(self, wrong: str, correct: str):
        """添加自定义词汇矫正"""
        self.unified_corrections[wrong] = correct
        logger.info(f"✅ 添加矫正: '{wrong}' -> '{correct}'")

    def batch_add_corrections(self, corrections_dict: Dict[str, str]):
        """批量添加词汇矫正"""
        self.unified_corrections.update(corrections_dict)
        logger.info(f"✅ 批量添加 {len(corrections_dict)} 个矫正")

    def remove_correction(self, word: str):
        """移除词汇矫正"""
        if word in self.unified_corrections:
            del self.unified_corrections[word]
            logger.info(f"✅ 移除矫正: '{word}'")
        else:
            logger.warning(f"⚠️ 词汇不存在: '{word}'")

    def get_vocabulary_stats(self) -> Dict[str, int]:
        """获取词汇表统计"""
        return {
            "total_corrections": len(self.unified_corrections)
        }

    def search_corrections(self, keyword: str) -> Dict[str, str]:
        """搜索相关矫正项"""
        results = {}
        for wrong, correct in self.unified_corrections.items():
            if keyword.lower() in wrong.lower() or keyword.lower() in correct.lower():
                results[wrong] = correct
        return results