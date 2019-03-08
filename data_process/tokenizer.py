import pandas as pd
import jieba

CLASSIFICATION_FILE_PATH = "classification.txt"
NEW_TRAIN_DATA_FILE_PATH = "../raw_data/newtrain.tsv"
WORD_TRAIN_DATA_FILE_PATH = "../data/word_train.tsv"

#建立分类名的索引表
item_type_index = {}
index = 0
with open(CLASSIFICATION_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        item_type_index[line] = index
        index += 1
print("共有分类名", len(item_type_index), "种")

#开始处理数据，将商品名称分词、商品分类名转换为索引
word_file = open(WORD_TRAIN_DATA_FILE_PATH, 'w', encoding='utf-8')
word_file.write('ITEM_NAME\tTYPE\n')

train_data = pd.read_csv(NEW_TRAIN_DATA_FILE_PATH, sep='\t', header=0)
print(len(train_data['ITEM_NAME']))
print(len(train_data['TYPE']))
for (item_name, item_type) in zip(train_data['ITEM_NAME'], train_data['TYPE']):
    item_name = item_name.strip()
    item_type = item_type.strip()
    seg_list = jieba.lcut(item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    item_type_int = item_type_index[item_type]
    word_file.write(item_name_str + '\t' + str(item_type_int) + '\n')

word_file.close()
