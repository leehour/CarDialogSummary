import pandas as pd
from gensim.models.word2vec import Word2Vec

from config import train_seg_path, test_seg_path, train_seg_merge_path, test_seg_merge_path, w2v_bin_path, \
    embedding_size, result_path


def build_dataset(data_train, data_test):
    lines = []
    for k in ['input', 'Report']:
        train_str = list(data_train[k].apply(str).values)
        if k != 'Report':
            test_str = list(data_test[k].apply(str).values)

        train_split = [i.split(' ') for i in train_str]
        test_split = [i.split(' ') for i in test_str]

        lines.extend(train_split)
        lines.extend(test_split)

    return lines


def build(train_vocab, w2v_bin_path="model.bin", embedding_size=256, min_count=5, col_sep='\t'):
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


if __name__ == '__main__':
    """open when you need train the w2v model
    
    """
    # data_train = pd.read_csv(train_seg_merge_path)
    # data_test = pd.read_csv(test_seg_merge_path)

    # train_texts = build_dataset(data_train, data_test)
    # print(len(train_texts))
    # build(train_texts, w2v_bin_path=w2v_bin_path, embedding_size=embedding_size, min_count=3)