# 文件结构

|- fwwb<br/>
|　　|- data　　　　　　　　　　　   # 预处理得到的数据<br/>
|　　|- data_process　　　　　　　 # 数据预处理代码<br/>
|　　|- glove　　　　　　　　　　   # glove预训练词向量代码<br/>
|　　|- models　　　　　　　　　　  # 模型代码<br/>
|　　|- raw_data　　　　　　　　　  # 比赛提供的原始数据<br/>
|　　|- WebVisualizaion　　　　　  # Web可视化代码<br/>

## raw_data文件夹

- test.tsv和train.tsv为赛题方提供的数据，由于系统原因我将文件的编码修改为utf-8，文件换行符修改为'\n',与Windows系统默认编码和换行符不同，使用PyCharm时应该无影响。<br/>
- newtrain.tsv为处理后的train.tsv，处理方式是扩充类别数量少的数据，具体方式为：若某类别的数据量少于700条，则扩充至700条；若数据量大于700条，则不做处理。<br/>

## data文件夹

- word_train.tsv是将newtrain.tsv预处理之后得到的文件，预处理方式为：1、使用jieba对商品名称进行分词 2、将商品类别转换为对应的整数标签。此文件用于模型训练。<br/>

## glove文件夹

此文件夹中其余文件不用管，只要注意两个文件即可，分别为corpus.txt和vectors.txt。<br/>
- corpus.txt文件是给glove训练用的语料，此语料的获取方式是将test.tsv和train.tsv中的所有商品名称分词后构成的语料。<br/>
- vectors.txt文件是glove使用语料训练之后生成的词向量文件。此文件用于模型训练。<br/>

## data_process文件夹（此文件夹为一个Pycharm项目）

此文件夹中的代码，1、先执行get_classification.py获取所有类别存储于classification.txt 2、再执行tokenizer.py，对newtrain.tsv进行处理，获得用于模型训练的word_train.tsv 3、get_corpus.py可忽略，不用执行。<br/>

- get_classification.py 此文件用于获取并排序训练数据newtrain.tsv中的所有类别名称，得到classification.txt文件。<br/>
- get_corpus.py 此文件用于获取glove所需的语料，语料的获取方式是将test.tsv和train.tsv中的所有商品名称分词后构成的语料。可忽略，不用执行。<br/>
- classification.txt 此文件保存了此次商品分类中所有商品的类别，共有1258种类别。<br/>
- tokenizer.py 此文件为将newtrain.tsv处理为word_train.tsv的代码，处理方式为：1、使用jieba对商品名称进行分词 2、将商品类别转换为对应的整数标签。<br/>

## models文件夹（此文件夹为一个Pycharm项目）

- model.py为模型训练代码，具体内容参见代码中的注释。<br/>
- predict.py（未完成）为使用模型进行商品分类预测的代码，具体内容参见代码中的注释。<br/>
- model文件夹用于存储模型训练所得到的分词器tokenizer.pickle以及模型的结构和参数model.h5。<br/>

## WebVisualizaion文件夹（Rails应用）

整个文件夹为使用Rails框架构建的Web应用，可忽略。<br/>
