import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

WORD_TRAIN_FILE_PATH = "../data/word_train.tsv"
TOKENIZER_PATH = "../data_process/additional_data/tokenizer.pickle"
GLOVE_EMBEDDING_PATH = "../glove/vectors.txt"
MODEL_WEIGHT_PATH = 'model_rnn/model_rnn.h5'
PLOT_MODEL_PATH = 'result_plot/model_rnn.png'

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
print("\n开始构建模型......")
maxlen = 25
max_words = 250000

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
sequences = tokenizer.texts_to_sequences(item_names)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = to_categorical(item_types)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 打乱数据顺序
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

#对嵌入进行预处理
embeddings_index = {}
f = open(GLOVE_EMBEDDING_PATH, 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# 准备GloVe词嵌入矩阵
embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 模型定义
category_num = 1258
recurrent_units_size = 300
dense_hidden_size = 4096
# Inputs
text_input = Input(shape=(maxlen,))

# Embeddings layers
embedded_text = layers.Embedding(max_words, embedding_dim, weights=[embedding_matrix])(text_input)

# Recurrent layers
# 1
bidirect_rnn_1 = layers.Bidirectional(
    layers.LSTM(recurrent_units_size, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
    merge_mode='concat'
)(embedded_text)
bidirect_rnn_2 = layers.Bidirectional(
    layers.LSTM(recurrent_units_size, dropout=0.5, recurrent_dropout=0.5),
    merge_mode='concat'
)(bidirect_rnn_1)

# 2
#bidirect_rnn = layers.Bidirectional(
#    layers.LSTM(recurrent_units_size, dropout=0.5, recurrent_dropout=0.5),
#    merge_mode='concat'
#)(embedded_text)

# 3
#bidirect_rnn = layers.Bidirectional(
#    layers.LSTM(recurrent_units_size, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
#    merge_mode='concat'
#)(embedded_text)
#rnn_layer = layers.LSTM(recurrent_units_size, dropout=0.5, recurrent_dropout=0.5)(bidirect_rnn)

# 4
#bidirect_rnn = layers.Bidirectional(
#    layers.LSTM(recurrent_units_size, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
#    merge_mode='concat'
#)(embedded_text)
#flatten_layer = layers.Flatten()(bidirect_rnn)

# Classifier
# 1
dense_1 = layers.Dense(dense_hidden_size, activation='relu')(bidirect_rnn_2)
# 2
#dense_1 = layers.Dense(dense_hidden_size, activation='relu')(bidirect_rnn)
# 3
#dense_1 = layers.Dense(dense_hidden_size, activation='relu')(rnn_layer)
# 4
#dense_1 = layers.Dense(dense_hidden_size, activation='relu')(flatten_layer)

batchNorm_3 = layers.BatchNormalization()(dense_1)
label_output = layers.Dense(category_num, activation='softmax')(batchNorm_3)
model = Model(text_input, label_output)

model.summary()
#plot_model(model, show_shapes=True, to_file=PLOT_MODEL_PATH)

# 添加模型回调函数
callback_list = [
    EarlyStopping(
        monitor='acc',
        patience=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_WEIGHT_PATH,
        monitor='val_loss',
        save_best_only=True,
    )
]

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(
    data, labels,
    epochs=30,
    batch_size=2048,
    callbacks=callback_list,
    validation_split=0.2
)

# 绘制结果图
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
