import random

import pandas
import config
import prepar_corpus.mylib.cut_sentences as cuts


def keywords_in_line(line):
    keyword_list = ""
    for word in line:
        if word in keyword_list:
            return True
        else:
            return False


def process_xiaohuangji(f_train, f_test):
    """
    处理小黄鸡的语料（chat）
    :return:
    """
    flags = [0, 0, 0, 0, 1]
    # 构建训练集和验证集的概率
    flag = 0
    for line in open(config.huangji_data_path, 'r', encoding='UTF-8').readlines():
        if line.startswith('E'):
            flag = 0
            continue
        elif line.startswith('M'):
            if flag == 0:
                line = line[1:].strip()
                flag = 1
            else:
                # 是否需要第二个M开头的句子
                continue

        line_cuted = cuts(line)
        if not keywords_in_line(line_cuted):
            line_cuted = " ".join(line_cuted) + "\t" + "__label__chat"
            if random.choice(flags) == 0:
                f_train.write(line_cuted + "\n")
            else:
                f_test.write(line_cuted + "\n")


# 将小黄鸡的语料调整为fasttext需要的格式，写到文件夹中
def process():
    f_train = open(config.classify_train_path, 'a', encoding='UTF-8')
    f_test = open(config.classify_test_path, 'a', encoding='UTF-8')
    process_xiaohuangji(f_train, f_test)
