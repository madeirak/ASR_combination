# ASR_combination

分别参考了两个开源项目，声学模型是cnn+ctc，语言模型是1-gram（不分词） 

对两个模型都有一定程度上的改进：
1-gram低频词间过渡处理(比如，“迷彩背包”的“彩背”，单个错误拼音如“mí cāi shū bāo”)


[声学模型](https://github.com/audier/DeepSpeechRecognition)
[语言模型](https://github.com/madeirak/ASRT_SpeechRecognition)
