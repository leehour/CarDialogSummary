import re

import pandas as pd
from gensim.models import KeyedVectors

from config import train_seg_merge_path, test_seg_merge_path, w2v_bin_path, checkpoint_dir, \
    result_path, embedding_size
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time
import os

from seq2seq_model import Encoder, Decoder

max_words_size = 30000
max_input_size = 500
max_target_size = 50
dataset_num = 100

open_bigru = False

EPOCHS = 10
BATCH_SIZE = 16
units = 128

params = {
    'learning_rate': 0.001,
    'adagrad_init_acc': 0.1,
    'max_grad_norm': 2
}

data_train_merge = pd.read_csv(train_seg_merge_path)
data_test_merge = pd.read_csv(test_seg_merge_path)

# load 词向量模型
model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)


def preprocess_word(w):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def tokenize(texts, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=max_words_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    tensor = tokenizer.texts_to_sequences(texts)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, max_len, padding='post')
    return tensor, word_index, tokenizer


def max_length(tensor):
    """
    :param tensor:
    :return:返回每个tensor的长度的最大值
    """
    return max(len(t) for t in tensor)


def get_embedding(word_index_input, word_index_target, embed_size=100):
    """
    获取input和target的embedding matrix
    :param word_index_input: 输入词典
    :param word_index_target: 输出词典
    :param embed_size:embedding size
    :return:
    """
    #     num_input_en = min(max_words_size, len(word_index_input)) + 1
    num_input_en = len(word_index_input) + 1
    encoder_embedding = np.zeros((num_input_en, embed_size))
    for word, id in word_index_input.items():
        if word not in model.vocab:
            word_vec = np.random.uniform(-0.25, 0.25, embed_size)
        else:
            word_vec = model.word_vec(word)
        encoder_embedding[id] = word_vec

    num_input_de = len(word_index_target) + 1
    #     num_input_de = min(max_words_size, len(word_index_target)) + 1
    decoder_embedding = np.zeros((num_input_de, embed_size))
    for word, id in word_index_target.items():
        if word not in model.vocab:
            word_vec = np.random.uniform(-0.25, 0.25, embed_size)
        else:
            word_vec = model.word_vec(word)
        decoder_embedding[id] = word_vec
    return encoder_embedding, decoder_embedding


def loss_function(real, pred, loss_object):
    """
    损失函数， 交叉熵损失
    :param real:
    :param pred:
    :return:
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden, loss_object, encoder, decoder, tokenizer_target, optimizer):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([tokenizer_target.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions, loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence, tokenizer_input, tokenizer_target, encoder, decoder, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_word(sentence)

    word_index = tokenizer_input.word_index
    inputs = [word_index[i] if i in word_index else word_index['<UNK>'] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    # hidden = [tf.zeros((1, 2 * units))]
    hidden = encoder.initialize_hidden_state(1)
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer_target.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out,
                                                             is_train=False)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tokenizer_target.index_word[predicted_id] + ' '

        if tokenizer_target.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def predict_report(sentence, tokenizer_input, tokenizer_target, encoder, decoder, max_length_inp, max_length_targ):
    return evaluate(sentence.lower(), tokenizer_input, tokenizer_target, encoder, decoder,
                    max_length_inp, max_length_targ)[0]


if __name__ == '__main__':
    # 添加<start><end>
    data_train_merge['input'] = data_train_merge['input'].apply(preprocess_word).copy()
    data_train_merge['Report'] = data_train_merge['Report'].apply(preprocess_word).copy()

    input_data = data_train_merge['input'].apply(str).values.tolist()
    target_data = data_train_merge['Report'].apply(str).values.tolist()

    tensor_input, word_index_input, tokenizer_input = tokenize(input_data, max_input_size)
    tensor_target, word_index_target, tokenizer_target = tokenize(target_data, max_target_size)

    max_length_targ, max_length_inp = max_length(tensor_target), max_length(tensor_input)

    # 构造embedding matrix
    encoder_embedding, decoder_embedding = get_embedding(word_index_input, word_index_target, embed_size=embedding_size)

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(tensor_input[:dataset_num], tensor_target[:dataset_num], test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = embedding_size
    vocab_inp_size = len(word_index_input) + 1
    vocab_tar_size = len(word_index_target) + 1

    # 构造训练数据集
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, encoder_embedding, open_bigru)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, decoder_embedding, open_bigru)

    optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state(BATCH_SIZE)
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, loss_object, encoder, decoder, tokenizer_target, optimizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # 此处预测训练集Reports
    predict_train_reports = []
    for sentence in data_train_merge['input'][:5]:
        result = predict_report(sentence, tokenizer_input, tokenizer_target, encoder, decoder, max_length_inp,
                                max_length_targ)
        predict_train_reports.append(result)
    print(predict_train_reports)

    # 此处预测测试集Reports
    data_test_merge['Report'] = data_test_merge[:20].apply(lambda x: predict_report(x['input'], tokenizer_input,
                                                                                    tokenizer_target, encoder, decoder,
                                                                                    max_length_inp,
                                                                                    max_length_targ), axis=1)
    # data_test_merge['Report'] = data_test_merge['input'][:100].apply(predict_report)
    data_test_merge.to_csv(result_path, index=False)