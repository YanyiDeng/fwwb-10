import pickle
import jieba
import numpy as np
import time
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils.np_utils import to_categorical


# 将预测结果的one_hot编码转换为对应的类别整数索引
def get_label_index(one_hot_label):
    return np.argmax(one_hot_label)


def jieba_tokenizer(temp_item_name):
    seg_list = jieba.lcut(temp_item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    return item_name_str


CLASSIFICATION_FILE_PATH = "../data_process/additional_data/classification.txt"
MODEL_INCEPTION_WEIGHT_PATH = 'model_inception/model_inception.h5'
MODEL_RNN_WEIGHT_PATH = 'model_rnn/model_rnn.h5'
TOKENIZER_PATH = "../data_process/additional_data/tokenizer.pickle"
DATA_PATH = "../raw_data/test.tsv"
RESULT_PATH = "../result/test_with_label.tsv"

# 计时
start = time.clock()

# 进行商品分类的预测
maxlen = 25

# 建立索引转分类名的列表
index_to_type = []
with open(CLASSIFICATION_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        index_to_type.append(line)

# 加载模型和分词器
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

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

text_input = Input(shape=(maxlen,))
inception_output = model_inception(text_input)
rnn_output = model_rnn(text_input)

models_list = [inception_output, rnn_output]

label_output = layers.average(models_list)
multi_model = Model(text_input, label_output)
multi_model.summary()

multi_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

# 获取需要预测的商品名称
item_names = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.rstrip()  #去除结尾的'\n'
        item_names.append(line)
f.close()
print("共有", len(item_names), "条数据等待打上标签")

# 处理商品名称
processed_item_names = [jieba_tokenizer(name) for name in item_names]
sequences = tokenizer.texts_to_sequences(processed_item_names)
data = pad_sequences(sequences, maxlen=maxlen)
# 预测商品标签
labels = multi_model.predict(data, batch_size=512)
item_types = []
for temp_label_one_hot in labels:
    label_index = get_label_index(temp_label_one_hot)
    label_index = int(label_index)
    item_types.append(index_to_type[label_index])
# 存入文件
with open(RESULT_PATH, 'w', encoding='utf-8') as f:
    f.write("ITEM_NAME\tTYPE\n")
    for (item_name, item_type) in zip(item_names, item_types):
        f.write(item_name + '\t' + item_type + '\n')

# 结束计时
elapsed = (time.clock() - start)
print("Time used:", elapsed)
