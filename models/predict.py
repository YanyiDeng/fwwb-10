import pickle
import jieba
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 将预测结果的one_hot编码转换为对应的类别整数索引
def get_label_index(one_hot_label):
    index = 0
    max_pro = 0
    label = -1
    for temp_value in one_hot_label:
        if temp_value > max_pro:
            max_pro = temp_value
            label = index
        index += 1
    return label


def jieba_tokenizer(item_name):
    seg_list = jieba.lcut(item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    return item_name_str


CLASSIFICATION_FILE_PATH = "../data_process/classification.txt"
MODEL_PATH = "model/model.h5"
TOKENIZER_PATH = "model/tokenizer.pickle"

# 建立索引转分类名的列表
index_to_type = []
with open(CLASSIFICATION_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        index_to_type.append(line)
print("总共有", len(index_to_type), "种分类名")

# 加载模型和分词器
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model(MODEL_PATH)
model.summary()

# 进行商品分类的预测
maxlen = 30
item_names = ["中国历史文选"]
processed_item_names = [jieba_tokenizer(name) for name in item_names]
print(processed_item_names)
sequences = tokenizer.texts_to_sequences(processed_item_names)
data = pad_sequences(sequences, maxlen=maxlen)
labels = model.predict(data)
for temp_name, temp_label_one_hot in zip(item_names, labels):
    label_index = get_label_index(temp_label_one_hot)
    print(temp_name, " --> ", index_to_type[label_index])
