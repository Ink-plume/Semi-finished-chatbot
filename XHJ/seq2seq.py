import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import chatbot.config as config


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)

    def forward(self,input_,target,input_len,target_len):
        encoder_outputs,encoder_hidden = self.encoder(input_,input_len)
        decoder_outputs,decoder_hidden = self.decoder(target,encoder_hidden)

        return decoder_outputs,decoder_hidden

    def evaluate(self,input_,input_len):
        encoder_outputs,encoder_hidden = self.encoder(input_,input_len)
        indices = self.decoder.evaluate(encoder_hidden)
        return indices