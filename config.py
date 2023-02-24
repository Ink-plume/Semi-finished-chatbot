import torch

user_dict_path = "../prepar_corpus/user_dict/user_dict.txt"

# 默认使用哈工大停用词表
stopwords_path = "../prepar_corpus/stopwords/hit_stopwords.txt"

huangji_data_path = "../prepar_corpus/data/xiaohuangji50w_nofenci.conv"

classify_train_path = "../prepar_corpus/classify/classify_train_data.txt"

classify_test_path = "../prepar_corpus/classify/classify_test_data.txt"

"""
分类相关
"""

classify_model_path = "../prepar_corpus/model/classify_model"

"""
XHJ路径
"""
XHJ_input_path = r"../data/XHJ_input.txt"
XHJ_target_path = r"../data/XHJ_target.txt"

XHJ_ws_input_path = r"../model/XHJ_ws_input_path.pkl"
XHJ_ws_target_path = r"../model/XHJ_ws_target_path.pkl"

"""
字典构建参数
"""

chatbot_batch_size = 256

chatbot_input_max_len = 20
chatbot_target_max_len = 15

"""
编码解码参数
"""

chatbot_embedding_dim = 256

"""
模型参数
"""
chatbot_encoder_num_layers = 1
chatbot_encoder_hidden_size = 128

chatbot_decoder_num_layers = 1
chatbot_decoder_hidden_size = 128

teacher_forcing_radio = 0.15

model_save_path = "../model/XHJ_seq2seq_model.model"
optimizer_save_path = "../model/XHJ_seq2seq_optimizer.model"

"""
设备参数
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")









