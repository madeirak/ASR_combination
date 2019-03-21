import numpy as np
import scipy.io.wavfile as wav
from utils import decode_ctc,compute_fbank
import tensorflow as tf
import os


from utils import get_data, data_hparams
data_args = data_hparams()
train_data = get_data(data_args)


speech_model_name = 'model_self.h5'
'''将测试文件放于test_wav/'''
test_wav_name = '5_.wav'


# 1.声学模型-----------------------------------
from Model_Speech import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('G:/ASR_combination/model_speech/' + speech_model_name)#从绝对路径的检查点恢复权重数据

#2.语言模型------------------------------------
from Model_Language import ModelLanguage

ml = ModelLanguage('model_language')
ml.LoadModel()

#3.翻译模型------------------------------
from trans_CN2EN.transformer import Graph
from trans_CN2EN.utils import create_hparams,en_segment,cn_segment,make_vocab
arg = create_hparams()
arg.is_training = False

#make_vocab
with open('trans_CN2EN/self.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()
    inputs = []
    outputs = []
    for line in (data[:20]):
        [cn, en] = line.strip('\n').split('\t')

        inputs.append(cn.replace(',', ' ,')[:-1].lower())  # 句中逗号后本有空格，在逗号前增加空格，然后将逗号按一个元素分隔，去掉句末标点，转为小写
        outputs.append(en[:-1])  # 去掉汉英语标签句末标点

    #print('分词前：', inputs[:10])
    #print('分词前：', outputs[:10])
    inputs = cn_segment(inputs)
    outputs = en_segment(outputs)
    #print('分词后：', inputs[:10])
    #print('分词后：', outputs[:10])
encoder_vocab,decoder_vocab = make_vocab(inputs,outputs)
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)



#4.声学模型测试--------------------------
import matplotlib.pyplot as plt
filepath = 'test_wav/'+test_wav_name

_, wavsignal = wav.read(filepath)
#plt.plot(wavsignal)
#plt.show()

fbank = compute_fbank(filepath)
#plt.imshow(fbank.T, origin = 'lower')
#plt.show()

pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))  #“//”整除，向下取整，“//”与“*”优先级相同，从左往右计算
                                                                #结果是a.shape[0]即每个元素的帧长可以被8整除

pad_fbank[:fbank.shape[0], :] = fbank
wav_data_lst = []
wav_data_lst.append(pad_fbank)

wav_lens = [len(data) for data in wav_data_lst]
wav_max_len = max(wav_lens)
new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
wav_lens = np.array([leng//8 for leng in wav_lens])

new_wav_data_lst[0, :wav_data_lst[0].shape[0], :, 0] = wav_data_lst[0]

#new_wav_data_lst = tf.expand_dims(new_wav_data_lst, 0)#3d->4d

result = am.model.predict(new_wav_data_lst, steps=1)#steps预测周期结束前的总步骤数(样品批次)，predict返回numpy数组类型的预测

_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
text = ' '.join(text)  # 以空格为分隔符合将多元素列表text合并成一个字符串
print('拼音结果：', text)


#5.测试语言模型------------------------------------

# pny2hanzi
text = text.split(' ')
#print(text)
str_pinyin = text
r = ml.SpeechToText(str_pinyin)
print('文字结果：', r)

#5.翻译模型测试----------------------
saver =tf.train.Saver()
with tf.Session() as sess:
    latest = tf.train.latest_checkpoint('trans_CN2EN/model_self')  # 查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest)  # restore(sess,save_path)，需要启动图表的会话。
    # 该save_path参数通常是先前从save()调用或调用返回的值latest_checkpoint()



    line = r
    #print(line[-1])
    #if line[-1] == ',' or '.' or '?' or '!'or'。'or'，'or'！':
    #    line = line[:-1]
    #print(line)
    line = line.replace('，', ' ,').strip('\n').split(' ')
    #print(line)
    line = cn_segment(line)
    #print(line)
    line = line[0]
    x = np.array([encoder_vocab.index(hanzi) for hanzi in line])
    x = x.reshape(1, -1)
    #print(x)
    de_inp = [[decoder_vocab.index('<GO>')]]  #de_inp  =  decoder_inputs
    while True:
        y = np.array(de_inp)
        preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
        #print(preds)
        if preds[0][-1] == decoder_vocab.index('<EOS>'):
            break
        de_inp[0].append(preds[0][-1])
    got = ' '.join(decoder_vocab[idx] for idx in de_inp[0][1:])
    print('英文结果:',got)