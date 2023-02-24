import fasttext
import config


def build_classify_model():
    model = fasttext.train_supervised(config.classify_corpus_path, epoch=20, lr=0.1, wordNgrams=1, minCount=5)
    model.save_model(config.classify_model_path)

def get_classify_model():
    model = fasttext.load_model(config.classify_model_path)
    return model

#  模型的评估
def eval():
    pass