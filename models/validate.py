import pickle
import pandas as pd
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "model_inception/model_inception.h5"
TOKENIZER_PATH = "../data_process/additional_data/tokenizer.pickle"
OLD_WORD_TRAIN_PATH = "../data/old_word_train.tsv"

# 加载模型和分词器
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model(MODEL_PATH)
model.summary()

test_item_names = []
test_item_types = []
old_word_train_data = pd.read_csv(OLD_WORD_TRAIN_PATH, sep='\t', header=0, low_memory=False)
print(len(old_word_train_data['ITEM_NAME']))
print(len(old_word_train_data['TYPE']))
for (item_name, item_type) in zip(old_word_train_data['ITEM_NAME'], old_word_train_data['TYPE']):
    item_name = item_name.strip()
    test_item_names.append(item_name)
    test_item_types.append(item_type)
print("商品名共有", len(test_item_names), "项")
print("商品分类共有", len(test_item_types), "项")

maxlen = 25
sequences = tokenizer.texts_to_sequences(test_item_names)
data = pad_sequences(sequences, maxlen=maxlen)
labels = to_categorical(test_item_types)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

cost = model.evaluate(data, labels, batch_size=4096)
print("loss:", cost[0], "  accuracy:", cost[1])
