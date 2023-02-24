import pickle
from torch.utils.data import DataLoader, Dataset
import chatbot.config as config
from word_sequence import Word_Sequence
import torch


class Chatbot_Dataset(Dataset):
    def __init__(self):
        self.input_path = config.XHJ_input_path
        self.target_path = config.XHJ_target_path
        self.input_lines = open(self.input_path, encoding='UTF-8').readlines()
        self.target_lines = open(self.target_path, encoding='utf-8').readlines()
        assert len(self.input_lines) == len(self.target_lines), "input和target长度一致"

    def __getitem__(self, index):
        input_item = self.input_lines[index].strip().split()
        target_item = self.target_lines[index].strip().split()
        input_length = len(input_item) if len(input_item) < config.chatbot_input_max_len else config.chatbot_input_max_len
        target_length = len(target_item) if len(target_item) < config.chatbot_target_max_len+1 else config.chatbot_target_max_len+1
        return input_item, target_item, input_length, target_length

    def __len__(self):
        return len(self.input_lines)


def collate_fn(batch):
    """
    :param batch:[(input_item,target_item,input_length,target_length),.......]四元 元组
    :return:
    排序
    """
    ws_input = pickle.load(open(config.XHJ_ws_input_path, 'rb'))
    ws_target = pickle.load(open(config.XHJ_ws_target_path, 'rb'))
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    input_item, target_item, input_length, target_length = zip(*batch)
    input_item = [ws_input.transform(i, max_len=config.chatbot_input_max_len) for i in input_item]
    input_item = torch.LongTensor(input_item)
    target_item = [ws_target.transform(i, max_len=config.chatbot_target_max_len, add_eos=True) for i in target_item]
    target_item = torch.LongTensor(target_item)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    # data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    # 使用自带的一键填充将填充到全部文本的最大值
    return input_item, target_item, input_length, target_length


train_data_loader = DataLoader(Chatbot_Dataset(), batch_size=config.chatbot_batch_size, shuffle=True,
                               collate_fn=collate_fn)
