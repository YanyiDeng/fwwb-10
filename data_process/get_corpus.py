import pandas as pd
import jieba

CORPUS_FILE_PATH = "../glove/corpus.txt"
TEST_DATA_FILE_PATH = "../raw_data/test.tsv"
TRAIN_DATA_FILE_PATH = "../raw_data/train.tsv"

corpus_file = open(CORPUS_FILE_PATH, 'w', encoding='utf-8')

test_data = pd.read_csv(TEST_DATA_FILE_PATH, sep='\t', header=0)
print(len(test_data['ITEM_NAME']))
for item_name in test_data['ITEM_NAME']:
    item_name = item_name.strip()
    seg_list = jieba.lcut(item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    corpus_file.write(item_name_str + '\n')

train_data = pd.read_csv(TRAIN_DATA_FILE_PATH, sep='\t', header=0)
print(len(train_data['ITEM_NAME']))
for item_name in train_data['ITEM_NAME']:
    item_name = item_name.strip()
    seg_list = jieba.lcut(item_name, cut_all=False)
    item_name_str = ' '.join(seg_list)
    corpus_file.write(item_name_str + '\n')

corpus_file.close()
