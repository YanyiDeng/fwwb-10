import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

WORD_VAL_FILE_PATH = "../data/word_val.tsv"
TOKENIZER_PATH = "../data_process/additional_data/tokenizer.pickle"
MODEL_CNN_WEIGHT_PATH = 'model_cnn/model_cnn.h5'
MODEL_INCEPTION_WEIGHT_PATH = 'model_inception/model_inception.h5'
MODEL_RNN_WEIGHT_PATH = 'model_rnn/model_rnn.h5'
MODEL_RCNN_WEIGHT_PATH = 'model_rcnn/model_rcnn.h5'

# 将验证数据和分类结果存入列表
val_names = []
val_types = []
word_val_data = pd.read_csv(WORD_VAL_FILE_PATH, sep='\t', header=0, low_memory=False)

for (val_name, val_type) in zip(word_val_data['ITEM_NAME'], word_val_data['TYPE']):
    val_name = val_name.strip()
    val_names.append(val_name)
    val_types.append(val_type)
print("验证商品名共有", len(val_names), "项")
print("验证商品分类共有", len(val_types), "项")

# 对数据的文本进行分词与建立索引
print("\n开始构建模型......")
maxlen = 25
max_words = 250000

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
val_sequences = tokenizer.texts_to_sequences(val_names)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

x_val = pad_sequences(val_sequences, maxlen=maxlen)

y_val = to_categorical(val_types)
print('Shape of val data tensor:', x_val.shape)
print('Shape of val label tensor:', y_val.shape)

# merge models

model_cnn = load_model(MODEL_CNN_WEIGHT_PATH)
model_cnn.name += '_cnn'
for layer in model_cnn.layers:
    layer.name += '_cnn'
    layer.trainable = False

model_inception = load_model(MODEL_INCEPTION_WEIGHT_PATH)
model_inception.name += '_inception'
for layer in model_inception.layers:
    layer.name += '_inception'
    layer.trainable = False

model_rnn = load_model(MODEL_RNN_WEIGHT_PATH)
model_rnn.name += '_rnn'
for layer in model_rnn.layers:
    layer.name += '_rnn'
    layer.trainable = False

model_rcnn = load_model(MODEL_RCNN_WEIGHT_PATH)
model_rcnn.name += '_rcnn'
for layer in model_rcnn.layers:
    layer.name += '_rcnn'
    layer.trainable = False

text_input = Input(shape=(maxlen,))
cnn_output = model_cnn(text_input)
inception_output = model_inception(text_input)
rnn_output = model_rnn(text_input)
rcnn_output = model_rcnn(text_input)

models_list = [cnn_output, inception_output]
#models_list = [cnn_output, rnn_output]
#models_list = [cnn_output, rcnn_output]

label_output = layers.average(models_list)
multi_model = Model(text_input, label_output)
multi_model.summary()

multi_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)
cost = multi_model.evaluate(x_val, y_val, batch_size=1024)
print("loss:", cost[0], "  accuracy:", cost[1])
