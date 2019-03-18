import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

#CLASSIFICATION_FILE_PATH = "../data_process/classification.txt"
WORD_TRAIN_FILE_PATH = "../data/word_train.tsv"

# 建立索引转分类名的列表
#index_to_type = []
#with open(CLASSIFICATION_FILE_PATH, 'r', encoding='utf-8') as f:
#    for line in f.readlines():
#        line = line.strip()
#        index_to_type.append(line)
#print(index_to_type[0], index_to_type[1257])
#print("总共有", len(index_to_type), "种分类名")

# 将训练数据和分类结果存入列表
item_names = []
item_types = []
word_train_data = pd.read_csv(WORD_TRAIN_FILE_PATH, sep='\t', header=0, low_memory=False)
#print(word_train_data['ITEM_NAME'][0], "=>", word_train_data['TYPE'][0])
#print(len(word_train_data['ITEM_NAME']))
#print(len(word_train_data['TYPE']))

for (item_name, item_type) in zip(word_train_data['ITEM_NAME'], word_train_data['TYPE']):
    item_name = item_name.strip()
    item_names.append(item_name)
    item_types.append(item_type)
print("商品名共有", len(item_names), "项")
print("商品分类共有", len(item_types), "项")
#print(item_names[:3])
#print(item_types[:3])
#print(item_names[-3:])
#print(item_types[-3:])
#print(item_names[23333])  # 检查第23334项，tsv中第23335行
#print(item_types[23333])

# 对数据的文本进行分词与建立索引
print("\n开始构建模型......")
maxlen = 30
max_words = 300000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(item_names)
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

# 保存分词索引
with open('model/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

#对嵌入进行预处理
GLOVE_EMBEDDING_PATH = "../glove/vectors.txt"
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
model = Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.SpatialDropout1D(0.5))
model.add(layers.Conv1D(512, 3, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(512, 3, activation='relu'))
model.add(layers.GRU(2048, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1258, activation='softmax'))

# 加载预训练GloVe嵌入
model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False
model.summary()

# 添加模型回调函数
callback_list = [
    EarlyStopping(
        monitor='acc',
        patience=1,
    ),
    ModelCheckpoint(
        filepath='model/model.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)
history = model.fit(
    data, labels,
    epochs=20,
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
