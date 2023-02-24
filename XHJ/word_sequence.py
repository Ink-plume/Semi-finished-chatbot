class Word_Sequence:
    PAD_TAG = "PAD"
    UNK_TAG = "TAG"
    SOS_TAG = "SOS"
    EOS_TAG = "EOS"
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS
        }
        self.count = {}

    def fit(self, sentence):
        """
        传入句子，进行词频统计
        :param sentence:str 句子
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=3, max_count=None, max_features=None):
        """
        构造词典
        :param min_count: 最低词数准入门槛
        :param max_count: 最大计数次数
        :param max_features: 词典总数量限制
        :return:
        """
        temp = self.count.copy()
        for key in temp:
            cur_count = self.count.get(key, 0)
            if min_count is not None:
                if cur_count < min_count:
                    del self.count[key]
            if max_count is not None:
                if cur_count > max_count:
                    del self.count[key]
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_features])
        for key in self.count:
            self.dict[key] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))



    def transform(self, sentence, max_len, add_eos=False):
        """
        把句子转换为数字序列
        :param sentence:str 句子
        :param max_len: 句子最大长度
        :param add_eos:是否添加结束标符
                        True 输出长度为max_len+1
                        False 输出长度为max_lan
        :return:
        """
        if len(sentence) > max_len:
            # 多余裁剪
            sentence = sentence[:max_len]
        sentence_len = len(sentence)
        if add_eos:
            sentence = sentence+[self.EOS_TAG]
        if sentence_len < max_len:
            # 空位填充
            sentence = sentence + [self.PAD_TAG]*(max_len-sentence_len)

        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def __len__(self):
        return len(self.dict)


    def inverse_transform(self, indices):
        result = []
        for i in indices:
            if i == self.EOS:
                break
            result.append(self.inverse_dict.get(i, self.UNK_TAG))
        return result
