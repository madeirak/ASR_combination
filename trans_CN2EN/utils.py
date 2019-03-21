from tqdm import tqdm
import jieba
import numpy as np
import tensorflow as tf


def create_hparams():
    params = tf.contrib.training.HParams(
        epochs=2,
        batch_size = 10,
        num_heads = 8,
        num_blocks = 6,

        input_vocab_size = 50,
        label_vocab_size = 50,

        max_length = 100, # embedding size
        hidden_units = 512,
        dropout_rate = 0.2,
        lr = 0.0003,
        is_training = True)
    return params


def en_segment(en_lst):
    inputs = [en.split(' ') for en in en_lst]
    return inputs

def cn_segment(cn_lst):
    cn_lst = [[char for char in jieba.cut(line) if char != ' '] for line in cn_lst]#提取jieba结果，丢弃空格
    return cn_lst



def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4):
    batch_num = len(encoder_inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        en_input_batch = encoder_inputs[begin:end]
        de_input_batch = decoder_inputs[begin:end]
        de_target_batch = decoder_targets[begin:end]
        max_en_len = max([len(line) for line in en_input_batch])
        max_de_len = max([len(line) for line in de_input_batch])
        en_input_batch = np.array([line + [0] * (max_en_len-len(line)) for line in en_input_batch])
        de_input_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_input_batch])
        de_target_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_target_batch])
        yield en_input_batch, de_input_batch, de_target_batch

def make_vocab(inputs,outputs):
    #make_vocab
    SOURCE_CODES = ['<PAD>']
    TARGET_CODES = ['<PAD>', '<GO>', '<EOS>']
    encoder_vocab = get_vocab(inputs, init=SOURCE_CODES)
    decoder_vocab = get_vocab(outputs, init=TARGET_CODES)
    return encoder_vocab,decoder_vocab


def get_vocab(data, init):
    #get_vocab_data
    vocab = init
    for line in tqdm(data):
        for word in line:
            if word not in vocab:
                vocab.append(word)
    return vocab


def data_format(inputs,outputs,encoder_vocab,decoder_vocab):
    # 调整数据格式，解码器输入起始部分有个开始符号，输出句尾有个结束符号
    encoder_inputs = [[encoder_vocab.index(word) for word in line] for line in inputs]
    decoder_inputs = [[decoder_vocab.index('<GO>')] + [decoder_vocab.index(word) for word in line] for line in outputs]
    decoder_targets = [[decoder_vocab.index(word) for word in line] + [decoder_vocab.index('<EOS>')] for line in
                       outputs]
    # print(decoder_inputs[:10])
    # print(decoder_targets[:10])
    return encoder_inputs,decoder_inputs,decoder_targets
