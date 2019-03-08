import pandas as pd

TRAIN_DATA_FILE_PATH = "../raw_data/train.tsv"
NEW_TRAIN_DATA_FILE_PATH = "../raw_data/newtrain.tsv"
WORD_TRAIN_DATA_FILE_PATH = '../data/word_train.tsv'
#train_data = pd.read_csv(TRAIN_DATA_FILE_PATH, sep='\t', header=0)
#print(len(train_data['ITEM_NAME']))
#print(len(train_data['TYPE']))
word_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH, sep='\t', header=0)
print(len(word_train_data['ITEM_NAME']))
print(len(word_train_data['TYPE']))
print('-------------')
num_of_category = {}
for temp_type in word_train_data['TYPE']:
    if temp_type in num_of_category:
        num_of_category[temp_type] += 1
    else:
        num_of_category[temp_type] = 1
print(num_of_category)
total_num = 0
for temp_value in num_of_category.values():
    total_num += temp_value
print(total_num)

print()
new_word_train_data = pd.read_csv(NEW_TRAIN_DATA_FILE_PATH, sep='\t', header=0)
print(len(new_word_train_data['ITEM_NAME']))
print(len(new_word_train_data['TYPE']))
print('-------------')
new_num_of_category = {}
for temp_type in new_word_train_data['TYPE']:
    if temp_type in new_num_of_category:
        new_num_of_category[temp_type] += 1
    else:
        new_num_of_category[temp_type] = 1
print(new_num_of_category)
new_total_num = 0
for temp_value in new_num_of_category.values():
    new_total_num += temp_value
print(new_total_num)