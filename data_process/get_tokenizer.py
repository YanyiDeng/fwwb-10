import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

WORD_TRAIN_FILE_PATH = "../data/word_train.tsv"

# 将训练数据和分类结果存入列表
item_names = []
item_types = []
word_train_data = pd.read_csv(WORD_TRAIN_FILE_PATH, sep='\t', header=0, low_memory=False)

for (item_name, item_type) in zip(word_train_data['ITEM_NAME'], word_train_data['TYPE']):
    item_name = item_name.strip()
    item_names.append(item_name)
    item_types.append(item_type)
print("商品名共有", len(item_names), "项")
print("商品分类共有", len(item_types), "项")

# 对数据的文本进行分词与建立索引
print("\n开始构建分词器......")
maxlen = 25
max_words = 250000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(item_names)
