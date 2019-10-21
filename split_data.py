import pandas as pd
import os
import jieba

from config import train_path, test_path, train_seg_path, test_seg_path, stop_words_path


# 停用词
def get_stop_words(stop_words_path):
    stop_words = '，：？。? ！! @ # $ % ^ & * ( （ ） ) [ ] { } > < = - + ~ ` --- (i (or / ; ;\' $1 |> \
                    --------- -------------------------------------------------------------------------- \
                    ========================= \
                    0 1 2 3 4 5 6 7 8 9 13 15 30 24 20 "a" tk> 95 45'
    with open(stop_words_path, 'r') as f:
        for line in f.readlines():
            stop_words = line + stop_words
    return stop_words


def process(s):
    stop_words = get_stop_words(stop_words_path)
    seg = [i for i in jieba.cut(s) if i not in stop_words]
    return " ".join(seg)


def build_vocab(df, sort=True, min_count=0, lower=False):
    data_columns = df.columns.tolist()
    data_seg = []
    df_new = pd.DataFrame()
    for col in data_columns:
        data_col = df[col]
        df[col] = df[col].apply(str)
        df_new[col] = df[col].apply(process)
    return df_new


if __name__ == '__main__':
    df_train = pd.read_csv(train_path, encoding='utf-8')
    df_test = pd.read_csv(test_path, encoding='utf-8')

    df_train_split = build_vocab(df_train)
    df_test_split = build_vocab(df_test)
    df_train_split.to_csv(train_seg_path, index=False)
    df_test_split.to_csv(test_seg_path, index=False)
