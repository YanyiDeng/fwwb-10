import random

TRAIN_FILE_PATH = "../raw_data/train.tsv"
TRAIN_8_FILE_PATH = "../raw_data/train_8.tsv"
TRAIN_2_FILE_PATH = "../raw_data/train_2.tsv"

with open(TRAIN_FILE_PATH, 'r', encoding='utf-8') as f:
  names_and_labels = f.readlines()

print(len(names_and_labels))

title = names_and_labels[0]
print(title)

names_and_labels = names_and_labels[1:]
print(len(names_and_labels))
print(names_and_labels[0])

# 打乱数据
random.shuffle(names_and_labels)
print(len(names_and_labels))
print(names_and_labels[0:3])

train_data = names_and_labels[0:400000]
val_data = names_and_labels[400000:]
print(len(train_data))
print(len(val_data))
print(train_data[0:3])
print(val_data[0:3])
print(len(train_data)+len(val_data))

with open(TRAIN_8_FILE_PATH, 'w', encoding='utf-8') as f:
    f.write(title)
    for temp_train_data in train_data:
        f.write(temp_train_data)

with open(TRAIN_2_FILE_PATH, 'w', encoding='utf-8') as f:
    f.write(title)
    for temp_val_data in val_data:
        f.write(temp_val_data)
