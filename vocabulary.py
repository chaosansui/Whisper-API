class Vocabulary:
    """词汇库管理类"""
    
    def __init__(self):
        # 粤语专用词汇表
        self.cantonese_corrections = {
            # 常用词汇
            "是不是": "係唔係",
            "这样": "咁樣",
            "那个": "嗰個",
            "这里": "呢度",
            "什么时候": "幾時",
            "为什么": "點解",
            "很好": "好好",
            "没有": "冇",
            "的": "嘅",
            "他": "佢",
            "我们": "我哋",
            "你们": "你哋",
            "他们": "佢哋",
            "这个": "呢個",
            "这些": "呢啲",
            "那些": "嗰啲",
            "哪里": "邊度",
            "什么": "咩",
            "怎么": "點樣",
            "多少": "幾多",
            "不行": "唔得",
            "可以": "得",
            "知道": "知",
            "不知道": "唔知",
            "谢谢": "唔該",
            "对不起": "對唔住",
            "现在": "而家",
            "刚才": "頭先",
            "明天": "聽日",
            "昨天": "琴日",
            "今天": "今日",
            
            # 金融客服相关
            "账户": "戶口",
            "密码": "密碼",
            "转账": "過數",
            "存款": "存款",
            "取款": "攞錢",
            "余额": "餘額",
            "贷款": "貸款",
            "利息": "利息",
            "信用卡": "信用卡",
            "投资": "投資",
            "股票": "股票",
            "基金": "基金",
            "保险": "保險",
            "理财": "理財",
            
            # 技术客服相关
            "登录": "登入",
            "注册": "登記",
            "验证": "驗證",
            "问题": "問題",
            "解决": "解決",
            "帮助": "幫手",
            "支持": "支援",
            "服务": "服務",
            "故障": "故障",
            "修复": "整返好",
            "设置": "設定",
            "更新": "更新",
            "下载": "下載",
            "安装": "安裝",
            "订单": "訂單",
            "支付": "付款",
            "发票": "發票",
            "退款": "退款",
            "优惠": "優惠",
            "会员": "會員",
            "积分": "積分",
            "配送": "送貨",
            "地址": "地址",
            "联系": "聯絡",
            "投诉": "投訴",
            "建议": "建議"
        }
        
        # 英文专用词汇表
        self.english_corrections = {
            # 常见拼写错误
            "there": "their",
            "your": "you're",
            "its": "it's",
            "to": "too",
            "then": "than",
            "weather": "whether",
            "accept": "except",
            "affect": "effect",
            "advice": "advise",
            "loose": "lose",
            "principal": "principle",
            
            # 客服场景专用
            "OK": "okay",
            "Punky": "Funky",
            "Punky Finance": "Funky Finance",
        }
        
        # 中文专用词汇表
        self.chinese_corrections = {
            "登陆": "登录",
            "帐号": "账号",
            "其它": "其他",
            "部份": "部分",
            "做业": "作业",
            "重起": "重启",
            "按装": "安装",
            "密妈": "密码"
        }
        
        self.industry_terms = {
            # 金融行业
            "mortage": "mortgage",
            "intrest": "interest",
            "withdrawl": "withdrawal",
            "deposite": "deposit",
            "transfered": "transferred",
            
        }
        
        
        self.colloquial_expressions = {
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of",
            "lemme": "let me",
            "gimme": "give me"
        }

    def get_corrections_for_language(self, language):
        """获取指定语言的词汇替换表"""
        if language == "Cantonese":
            return self.cantonese_corrections
        elif language == "English":
            return {**self.english_corrections, **self.industry_terms, **self.colloquial_expressions}
        elif language == "Chinese":
            return self.chinese_corrections
        else:
            return {}

    def add_custom_vocabulary(self, language, corrections_dict):
        """添加自定义词汇"""
        if language == "Cantonese":
            self.cantonese_corrections.update(corrections_dict)
        elif language == "English":
            self.english_corrections.update(corrections_dict)
        elif language == "Chinese":
            self.chinese_corrections.update(corrections_dict)

    def remove_vocabulary(self, language, word):
        """移除词汇"""
        if language == "Cantonese" and word in self.cantonese_corrections:
            del self.cantonese_corrections[word]
        elif language == "English" and word in self.english_corrections:
            del self.english_corrections[word]
        elif language == "Chinese" and word in self.chinese_corrections:
            del self.chinese_corrections[word]