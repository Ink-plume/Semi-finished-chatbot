import chatbot.config as config
import string
import jieba
from tqdm import tqdm
import jieba.posseg as psg


def filter(pair):
    """
    过滤
    :param pair:
    :return:
    """
    if pair[0][1].strip() in list(string.ascii_lowercase):
        return True
    elif pair[0][1].strip() in list(string.ascii_uppercase):
        return True
    elif pair[1][1].count("=") >= 2 and len(pair[1][0].split())<4:
        return True
    elif "黄鸡" in pair[0][1] or "黄鸡" in pair[1][1] or "小通" in pair[0][1] or "小通" in pair[1][1]:
        return True
    elif len(pair[0][0].strip()) == 0 or len(pair[1][0].strip()) == 0:
        return True


def process_xiaohuangji():
    """
    处理小黄鸡的语料（chat）
    :return:
    """
    input_path = config.XHJ_input_path
    target_path = config.XHJ_target_path
    data_path = config.huangji_data_path
    f_input = open(input_path, 'a', encoding='utf-8')
    f_target = open(target_path, 'a', encoding='utf-8')

    one_qa_pair = []
    num = 0
    for line in tqdm(open(data_path, 'r', encoding='UTF-8').readlines(), ascii=True, desc="处理小黄鸡"):
        if line.startswith('E'):
            continue
        else:
            line = line[1:].strip()
            line_cut = cut_sentence(line)
            line_cut = " ".join(line_cut)
        if len(one_qa_pair)<2:
            one_qa_pair.append([line_cut,line])
        if len(one_qa_pair) == 2:
            if filter(one_qa_pair):
                one_qa_pair = []
                continue
            f_input.write(one_qa_pair[0][0]+'\n')
            f_target.write(one_qa_pair[1][0]+'\n')
            num+=1
            one_qa_pair = []

    f_input.close()
    f_target.close()


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
        # 是否使用停用词
        if stopword:
            result = [i for i in result if i not in stopword]
    return result


if __name__ == '__main__':
    process_xiaohuangji()
