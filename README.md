# ASR_combination

分别参考了两个开源项目，声学模型是cnn+ctc，语言模型是1-gram（不分词） 

对两个模型都有一定程度上的改进：
1-gram超低频词间过渡处理(比如，“迷彩背包”的“彩背”，声学模型单个错误拼音如“mí cāi shū bāo”)

声学模型的输出是“kǎo yán yān yǔ cí huì”  时，因为“yān”会使连同之前的三个拼音的联合概率过低，造成丢弃，只输出“语词汇”，造成理解困难。
修改后的1-gram会输出“ 考研烟语词汇 ”，不会造成词语的丢失。



[参考声学模型](https://github.com/audier/DeepSpeechRecognition)
[参考语言模型](https://github.com/madeirak/ASRT_SpeechRecognition)
