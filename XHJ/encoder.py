from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import chatbot.config as config
import pickle


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ws_input = pickle.load(open(config.XHJ_ws_input_path, 'rb'))
        self.ws_target = pickle.load(open(config.XHJ_ws_target_path, 'rb'))
        self.embedding = nn.Embedding(num_embeddings=len(self.ws_input),
                                      embedding_dim=config.chatbot_embedding_dim,
                                      padding_idx=self.ws_input.PAD)
        """
        num_embeddings 字典的大小
        embedding_dim 将word编为多少维的向量
        padding_inx 填充索引，如果是该索引，编码为0
        unknown 如果是未知单词，编码为0
        """
        self.gru = nn.GRU(input_size=config.chatbot_embedding_dim,
                          num_layers=config.chatbot_encoder_num_layers,
                          hidden_size=config.chatbot_encoder_hidden_size,
                          batch_first=True
                          )
        """
        input_size：x的特征维度
        hidden_size：隐藏层的特征维度
        num_layers：lstm隐层的层数，默认为1
        bias：False则bih=0和bhh=0. 默认为True
        batch_first：True则输入输出的数据格式为 (batch, seq, feature)
        dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
        bidirectional：True则为双向lstm默认为False
        """

    def forward(self, input_, input_len):
        embeded = self.embedding(input_)

        embeded = pack_padded_sequence(embeded, input_len, batch_first=True)
        """
        input：经过处理之后的数据。
        lengths：batch中各个序列的实际长度。
        batch_first：True 对应 [batch_size, seq_len, feature] ；False 对应 [seq_len, batch_size, feature] 。
        enforce_sorted：如果是 True ，则输入应该是按长度降序排序的序列。如果是 False ，会在函数内部进行排序。默认值为 True 。  
        """
        out, hidden = self.gru(embeded)

        out, out_len = pad_packed_sequence(out, batch_first=True, padding_value=self.ws_input.PAD)

        return out,hidden
    