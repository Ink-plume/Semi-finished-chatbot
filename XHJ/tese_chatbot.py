import chatbot.config as config
from chatbot.XHJ.word_sequence import Word_Sequence
import pickle
from dataset import train_data_loader
from train import train


def save_ws():
    ws = Word_Sequence()
    for line in open(config.XHJ_input_path,encoding='UTF-8').readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(config.XHJ_ws_input_path,'wb'))

    ws = Word_Sequence()
    for line in open(config.XHJ_target_path,encoding='UTF-8').readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(config.XHJ_ws_target_path,'wb'))


def test_data_loader():
        i = 0
        for idx, (input,target,input_len,target_len)in enumerate(train_data_loader):
            print("\n\n")
            print(idx)
            print(input)
            print(target)
            print(input_len)
            print(target_len)
            print("\n\n")
            i+=1
            if i == 3 :
                break


def train_seq2seq():
    for i in range(10):
        train(i)


if __name__ == '__main__':
    train_seq2seq()