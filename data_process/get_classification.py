import pandas as pd

TRAIN_DATA_FILE_PATH = '../raw_data/train.tsv'
CLASSIFICATION_FILE_PATH = 'additional_data/classification.txt'

classification_set = set()  # 用于存储所有分类名的集合

# 获取所有分类名
train_data = pd.read_csv(TRAIN_DATA_FILE_PATH, sep='\t', header=0)
for temp_type in train_data['TYPE']:  # 获取测试数据中的每个分类名
    type_str = temp_type.strip()
    if type_str not in classification_set:  # 如果分类名集合中不存在当前分类名，则添加到集合
        classification_set.add(type_str)

print('总共有', len(classification_set), '种分类名')

# 将分类名写入txt文件
classification_list = list(classification_set)
classification_list.sort()
with open(CLASSIFICATION_FILE_PATH, 'w', encoding='utf-8') as f:
    for temp_type_str in classification_list:
        temp_output_str = temp_type_str + '\n'
        f.write(temp_output_str)
