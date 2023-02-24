import torch.nn as nn
import torch.nn.functional as F
import chatbot.config as config
import random
import torch
import pickle


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ws_input = pickle.load(open(config.XHJ_ws_input_path, 'rb'))
        self.ws_target = pickle.load(open(config.XHJ_ws_target_path, 'rb'))
        self.embedding = nn.Embedding(num_embeddings=len(self.ws_target),
                                      embedding_dim=config.chatbot_embedding_dim,
                                      padding_idx=self.ws_target.PAD
                                      )
        self.gru = nn.GRU(input_size=config.chatbot_embedding_dim,
                          hidden_size=config.chatbot_decoder_hidden_size,
                          num_layers=config.chatbot_decoder_num_layers,
                          batch_first=True)

        # 将隐藏层升维映射到词典的形状上
        self.fc = nn.Linear(config.chatbot_decoder_hidden_size, len(self.ws_target))

    def forward(self, target, encoder_hidden, teacher_force=True):
        # 获取encoder的输出作为decoder的初始hidden_State
        decoder_hidden = encoder_hidden
        batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * self.ws_target.SOS).to(
            config.device)
        decoder_outputs = torch.zeros([batch_size, config.chatbot_target_max_len + 1, len(self.ws_target)]).to(
            config.device)

        if teacher_force:
            if random.random() < config.teacher_forcing_radio:
                for t in range(config.chatbot_target_max_len + 1):
                    decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                    # 保存每一步的output存入outputs，方便计算loss
                    decoder_outputs[:, t, :] = decoder_output_t
                    decoder_input = target[:, t].unsqueeze(-1)
            else:
                for t in range(config.chatbot_target_max_len + 1):
                    decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                    # 保存每一步的output存入outputs，方便计算loss
                    decoder_outputs[:, t, :] = decoder_output_t
                    value, index = torch.topk(decoder_output_t, 1)
                    decoder_input = index
        else:
            for t in range(config.chatbot_target_max_len + 1):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                # 保存每一步的output存入outputs，方便计算loss
                decoder_outputs[:, t, :] = decoder_output_t
                value, index = torch.topk(decoder_output_t, 1)
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        """
        计算每个时间步上的结果
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return:
        """
        decoder_input_embeded = self.embedding(decoder_input)  # [batch_size,1,embedding_dim]
        out, decoder_hidden = self.gru(decoder_input_embeded, decoder_hidden)
        """
        out: [bat_s,1,hid_s]
        decoder_hidden: [1 ,bat_s ,hid_s]
        """
        out = out.squeeze(1)
        output = F.log_softmax(self.fc(out), dim=-1)

        return output, decoder_hidden

    def evaluate(self, encoder_hidden):
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(0)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * self.ws_target.SOS).to(
            config.device)

        indices = []

        for i in range(config.chatbot_target_max_len + 10):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index
            indices.append(index.squeeze(-1).cpu().detach().numpy())
        return indices
