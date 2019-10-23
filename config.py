import os

# stop_words 路径
stop_words_path = os.path.join(os.path.abspath('./'), 'datasets', 'stopwords.txt')

# 数据路径
train_path = os.path.join(os.path.abspath('./'), 'datasets', 'AutoMaster_TrainSet.csv')
test_path = os.path.join(os.path.abspath('./'), 'datasets', 'AutoMaster_TestSet.csv')

# 切分词之后保存文件
train_seg_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg.csv')
test_seg_path = os.path.join(os.path.abspath('./'), 'datasets', 'test_seg.csv')

# 合并列之后保存文件
train_seg_merge_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg_merge.csv')
test_seg_merge_path = os.path.join(os.path.abspath('./'), 'datasets', 'test_seg_merge.csv')

# Word2Vec模型存放路径
w2v_bin_path = os.path.join(os.path.abspath('./'), 'model', 'model.bin')

# checkpoints 存储路径
checkpoint_dir = os.path.join(os.path.abspath('./'), 'checkpoints', 'training_checkpoints')

# result path
result_path = os.path.join(os.path.abspath('./'), 'datasets', 'result.csv')

embedding_size = 256
