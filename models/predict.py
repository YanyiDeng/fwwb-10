import pickle
import jieba
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# 将预测结果的one_hot编码转换为对应的类别整数索引
def get_label_index(one_hot_label):
    return np.argmax(one_hot_label)


def jieba_tokenizer(item_name):
    seg_list = jieba.lcut(item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    return item_name_str


CLASSIFICATION_FILE_PATH = "../data_process/additional_data/classification.txt"
MODEL_PATH = "model/model.h5"
TOKENIZER_PATH = "../data_process/additional_data/tokenizer.pickle"
RESULT_PATH = "../WebVisualization/lib/assets/result.txt"

# 建立索引转分类名的列表
index_to_type = []
with open(CLASSIFICATION_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        index_to_type.append(line)

# 加载模型和分词器
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model(MODEL_PATH)

# 进行商品分类的预测
maxlen = 30
# 获取命令行参数中需要预测的商品名称
param = sys.argv[1]
item_names = param.split('&&')
processed_item_names = [jieba_tokenizer(name) for name in item_names]
sequences = tokenizer.texts_to_sequences(processed_item_names)
data = pad_sequences(sequences, maxlen=maxlen)
labels = model.predict(data)
item_types = []
for temp_label_one_hot in labels:
    label_index = get_label_index(temp_label_one_hot)
    label_index = int(label_index)
    item_types.append(index_to_type[label_index])
with open(RESULT_PATH, 'w') as f:
    f.write('&&'.join(item_types))
