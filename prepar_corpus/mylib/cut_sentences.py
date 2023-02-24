import jieba
import chatbot.config as config
import string
import jieba.posseg as psg


# jieba.load_userdict(config.user_dict_path)


letters = string.ascii_lowercase + string.ascii_uppercase


# 实现中英文分词
def cut_sentence_ch_and_en_byword(sentence):
    result = []
    temp = ""
    for word in sentence:
        if word in letters:
            temp += word
        else:
            if temp != "":
                result.append(temp)
                temp = ""
                result.append(word.strip())
            else:
                result.append(word.strip())
    # 如果最后的字符是英文
    if temp != "":
        result.append(temp)
    return result


def cut_sentence(sentence, by_word=False, stopword=False, sg=False):
    """
    :param sentence:str 传入句子
    :param by_word:是否按照单个字分词
    :param stopword:是否使用停用词
    :param sg:是否返回词性
    :return:
    """
    # 是否按单个字返回
    if by_word:
        result = cut_sentence_ch_and_en_byword(sentence)
    else:
        # 是否返回词性
        if sg:
            result = psg.lcut(sentence)
            result = [(i.word, i.flag) for i in result]
        else:
            result = jieba.lcut(sentence)
        #是否使用停用词
        if stopword:
            result = [i for i in result if i not in stopword]
    return result


if __name__ == '__main__':
    sentence = "Wang腾麒是不是猪haha"
    print(cut_sentence_ch_and_en_byword(sentence))
