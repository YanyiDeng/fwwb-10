# 文件结构

|- fwwb<br/>
|　　|- data　　　　　　　　　　　   # 预处理得到的数据<br/>
|　　|- data_process　　　　　　　 # 数据预处理代码<br/>
|　　|- glove　　　　　　　　　　   # glove预训练词向量代码<br/>
|　　|- models　　　　　　　　　　  # 模型代码<br/>
|　　|- raw_data　　　　　　　　　  # 比赛提供的原始数据<br/>
|　　|- result　　　　　　　　　    # 通过模型对test.tsv和train.tsv打标签后的结果<br/>
|　　|- WebVisualizaion　　　　　  # Web可视化代码<br/>

## raw_data文件夹

- test.tsv和train.tsv为赛题方提供的数据。<br/>
- train_2.tsv和train_8.tsv是train.tsv按照每个类别随机取80%和20%得到的划分数据。<br/>
- newtrain.tsv为处理后的train_8.tsv，处理方式是扩充类别数量少的数据，具体方式为：若某类别的数据量少于700条，则扩充至700条；若数据量大于700条，则不做处理。<br/>

## data文件夹

- word_train.tsv是将newtrain.tsv预处理之后得到的文件，预处理方式为：1、使用jieba对商品名称进行分词 2、将商品类别转换为对应的整数标签。此文件用于模型训练。<br/>
- word_val.tsv是将train_2.tsv预处理之后得到的文件，预处理方式同上。此文件用于验证模型精度。<br/>
- old_word_train.tsv是train.tsv预处理之后得到的文件，预处理方式同上。此文件用于验证模型在原本的50万训练数据上的精度。<br/>

## glove文件夹

此文件夹中其余文件为glove原有代码，只要注意两个文件即可，分别为corpus.txt和vectors.txt。<br/>
- corpus.txt文件是给glove训练用的语料，此语料的获取方式是将test.tsv和train.tsv中的所有商品名称分词后构成的语料。<br/>
- vectors.txt文件是glove使用语料训练之后生成的词向量文件。此文件用于模型训练。<br/>

## data_process文件夹

- divide_data.cpp为将train.tsv根据每个类别随机取80%和20%数据形成train_8.tsv和train_2.tsv的代码。<br/>
- extend_data.cpp为将train_8.tsv进行数据扩充的代码，形成newtrain.tsv。<br/>
- get_classification.py此文件用于获取并排序训练数据train.tsv中的所有类别名称，得到classification.txt文件。<br/>
- get_corpus.py此文件用于获取glove所需的语料，语料的获取方式是将test.tsv和train.tsv中的所有商品名称分词后构成的语料。<br/>
- get_tokenizer.py此文件用于获取keras中tokenizer分词器的单词索引字典的相关参数，并存于additional_data/tokenizer.pickle。<br/>
- process_train_data.py此文件为将newtrain.tsv处理为word_train.tsv、将train.tsv处理为old_word_train.tsv、将train_2.tsv处理为word_val.tsv的代码，处理方式为：1、使用jieba对商品名称进行分词 2、将商品类别转换为对应的整数标签。<br/>

## models文件夹

- model_xxx.py为模型xxx的对应代码。相应的文件夹存储模型xxx的结构以及权重。MultiModel.py为模型融合的代码。<br/>
- predict.py为使用模型进行商品分类预测的代码。<br/>
- validate.py为使用模型对原本的50万训练数据进行验证的代码。<br/>
- classify_test.py和classify_train.py分别为对test.tsv和train.tsv中的商品名称进行分类预测并打上标签的代码。<br/>
- result_plot文件夹存储各个模型的结构图以及训练时的损失和精度曲线图。<br/>

## result文件夹

test_with_label.tsv和train_with_label.tsv分别为对test.tsv和train.tsv中的商品名称进行分类预测并打上标签后的结果。<br/>

## WebVisualizaion文件夹（Rails应用）

整个文件夹为使用Rails框架构建的Web应用。<br/>
