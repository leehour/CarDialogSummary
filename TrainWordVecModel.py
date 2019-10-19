import pandas as pd
from gensim.models.word2vec import Word2Vec

from Config import train_seg_path, test_seg_path, train_seg_merge_path, test_seg_merge_path


def build_dataset(data_train, data_test):
    lines = []
    for k in ['Brand', 'Model', 'Question', 'Dialogue', 'Report']:
        train_str = list(data_train[k].apply(str).values)
        if k != 'Report':
            test_str = list(data_test[k].apply(str).values)

        train_split = [i.split(' ') for i in train_str]
        test_split = [i.split(' ') for i in test_str]

        lines.extend(train_split)
        lines.extend(test_split)

    return lines


def build(train_vocab, out_path=None, embedding_size=100, sentence_path='',
          w2v_bin_path="model.bin", min_count=5, col_sep='\t'):
    #     sentences = extract_sentence(train_seg_path, test_seg_path, col_sep=col_sep)
    #     save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=train_vocab,
                   size=embedding_size, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('奔驰', '宝马')
    print('奔驰 vs 宝马 similarity score:', sim)


data_train = pd.read_csv(train_seg_path)
data_test = pd.read_csv(test_seg_path)

# 训练词向量模型时打开
# train_texts = build_dataset(data_train, data_test)
# print(len(train_texts))
# build(train_texts, w2v_bin_path=w2v_bin_path, min_count=3)

data_train.dropna(axis=0, how='any', inplace=True)
data_test.dropna(axis=0, how='any', inplace=True)

# 合并除report的字段
data_train['input'] = data_train['Brand'] + ' ' + data_train['Model'] + ' ' + data_train['Question'] + ' ' + data_train[
    'Dialogue']
data_train.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)

data_test['input'] = data_test['Brand'] + ' ' + data_test['Model'] + ' ' + data_test['Question'] + ' ' + data_test[
    'Dialogue']
data_test.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)

data_train.to_csv(train_seg_merge_path, index=False)
data_test.to_csv(test_seg_merge_path, index=False)
