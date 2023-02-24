import torch.nn as nn
import torch.nn.functional as F
import chatbot.config as config
import torch
from tqdm import tqdm
from torch.optim import Adam
from dataset import train_data_loader
from seq2seq import Seq2Seq
import pickle


ws_input = pickle.load(open(config.XHJ_ws_input_path, 'rb'))
ws_target = pickle.load(open(config.XHJ_ws_target_path, 'rb'))

seq2seq = Seq2Seq()
seq2seq = seq2seq.to(config.device)
optimizer = Adam(seq2seq.parameters(),lr=0.001)

def train(epoch):
    bar = tqdm(enumerate(train_data_loader),total=len(train_data_loader),ascii=True,desc='train')

    for idx,(input_,target,input_len,target_len) in bar:

        input_ = input_.to(config.device)
        target = target.to(config.device)
        input_len = input_len.to(config.device)
        target_len = target_len.to(config.device)

        optimizer.zero_grad() # 梯度置为0
        decoder_outputs,_ = seq2seq(input_,target,input_len,target_len)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1),-1)  # [batch_size*seq_len,-1]
        target = target.view(-1)
        loss = F.nll_loss(decoder_outputs,target,ignore_index=ws_target.PAD)
        loss.backward()
        optimizer.step()
        bar.set_description("epoch:{}\tidx:{}\tloss:{:.4f}".format(epoch,idx,loss.item()))
        if idx % 400 == 0:
            torch.save(seq2seq.state_dict(),config.model_save_path)
            torch.save(optimizer.state_dict(),config.optimizer_save_path)


