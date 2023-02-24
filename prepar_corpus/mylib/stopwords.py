"""
获取停用词
"""

import chatbot.config as config

stopword = [i.strip() for i in open(config.stopwords_path, 'r', encoding="UTF-8").readlines()]
