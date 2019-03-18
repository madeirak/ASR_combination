#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance


# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
train_data = get_data(data_args)


# 1.声学模型-----------------------------------
from Model_Speech import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('model_speech/model_self.h5')#从检查点恢复权重数据

#2.语言模型------------------------------------
from Model_Language import ModelLanguage

ml = ModelLanguage('model_language')
ml.LoadModel()



# 3. 准备测试所需数据， 通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'train'
data_args.shuffle = False
data_args.batch_size = 1
data_args.self_wav = True
test_data = get_data(data_args)

# 4. 进行测试-------------------------------------------
am_batch = test_data.get_am_batch()
word_num = 0
word_error_num = 0
for i in range(10):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, _ = next(am_batch)#yield inputs, outputs
    x = inputs['the_inputs']#即x = pad_wav_data
    y = test_data.pny_lst[i]
    result = am.model.predict(x, steps=1)#steps预测周期结束前的总步骤数(样品批次)，predict返回numpy数组类型的预测

    # 将数字结果转化为pny结果
    _, text = decode_ctc(result, train_data.am_vocab)#num2pny
    text = ' '.join(text)#以空格为分隔符合将多元素列表text合并成一个字符串
    print('文本结果：', text)
    print('原文结果：', ' '.join(y))#以空格为分隔符将多元素列表y合并成一个字符串


    #pny2hanzi
    text = text.split(' ')
    str_pinyin = text
    r = ml.SpeechToText(str_pinyin)
    print('文字结果：',r)