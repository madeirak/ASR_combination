#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
语音识别的语言模型

基于马尔可夫模型的语言模型

"""

class ModelLanguage(): # 语音模型类
	def __init__(self, modelpath):
		self.modelpath = modelpath
		self.slash = '\\'

		if(self.slash != self.modelpath[-1]): # 在目录路径末尾增加斜杠
			self.modelpath = self.modelpath + self.slash
		
		pass
		
	def LoadModel(self):
		self.dict_pinyin = self.GetSymbolDict('dict.txt')   #dict_pinyin---pny2word
		self.model1 = self.GetLanguageModel(self.modelpath + 'language_model1.txt') #读取单字词词频统计文件，返回字典类型
		self.model2 = self.GetLanguageModel(self.modelpath + 'language_model2.txt') #读取双字词词频统计文件，返回字典类型
		self.pinyin = self.GetPinyin(self.modelpath + 'dic_pinyin.txt')
		model = (self.dict_pinyin, self.model1, self.model2 )
		return model
		pass
	
	def SpeechToText(self, list_syllable):  #pny2word
		'''
		为语音识别专用的处理函数
		实现从语音拼音符号到最终文本的转换
		'''
		r=''
		length = len(list_syllable)
		if(length == 0): # 传入的参数没有包含任何拼音时
			return ''
		
		# 先取出一个字，即拼音列表中第一个pny
		str_tmp = [list_syllable[0]]
		
		for i in range(0, length - 1):
			# 依次从第一个字开始每次连续取两个字拼音
			str_split = list_syllable[i] + ' ' + list_syllable[i+1]
			#print(str_split,str_tmp,r)

			'''1-gram'''
			if(str_split in self.pinyin): # 如果这包含2pny的拼音组在汉语拼音状态转移字典里的话(dic_pinyin.txt)
				# 将第二个字的拼音加入
				str_tmp.append(list_syllable[i+1])
			else:
				# 否则不加入，然后直接将现有的拼音序列进行解码
				str_decode = self.decode(str_tmp, 0.0000)
				#print(str_decode)
				#print('decode ',str_tmp,str_decode)
				if(str_decode != []):
					r += str_decode[0][0]

				# 再重新从i+1开始作为第一个拼音
				str_tmp = [list_syllable[i+1]]
				
		
		#print('最后：', str_tmp)
		str_decode = self.decode(str_tmp, 0.0000)
		
		#print('剩余解码：',str_decode)
		
		if(str_decode != []):
			r += str_decode[0][0]

		return r
	
	def decode(self,list_syllable, yuzhi = 0.0001):    #syllable 音节  list_syllable为需解码的一个pny或两个pny
		'''
		实现拼音向文本的转换
		基于马尔可夫链
		'''
		#assert self.dic_pinyin == null or self.model1 == null or self.model2 == null
		list_words = []
		
		num_pinyin = len(list_syllable)
		#print('======')
		#print('decode function: list_syllable\n',list_syllable)
		#print(num_pinyin)

		# 开始语音解码
		for i in range(num_pinyin): #遍历传入的pny列表 （num_pinyin = 1 或 2）
			#print(i)
			ls = ''
			if(list_syllable[i] in self.dict_pinyin): # 如果这个拼音在拼音字典里的话

				# 获取拼音下属的同音异字字的列表
				ls = self.dict_pinyin[list_syllable[i]]  #ls包含了该拼音对应的所有的字
			else:
				break
			
			
			if(i == 0):  #如果解码的是两个pny中的第一个pny，或只传入了一个pny
				# 第一个字做初始处理
				num_ls = len(ls)  #该pny同音异字个数
				for j in range(num_ls): #遍历同音异字
					#tuple_word = ['',0.0]

					# 设置马尔科夫模型初始状态值
					# 设置初始概率，置为1.0
					tuple_word = [ls[j], 1.0]
					#print(tuple_word)
					# 添加到可能的句子列表
					list_words.append(tuple_word)  #把该pny所有同音异字以二元组的形式添加入list_words，每个二元组tuple_word（ 字 ，概率 ）
				
				#print(list_words)
				continue
			else:
				# 开始处理紧跟在第一个字后面的字
				list_words_2 = []
				num_ls_word = len(list_words)    #当前list_words的长度
				#print('ls_wd: ',list_words)
				for j in range(0, num_ls_word):  #遍历之前预测的所有候选字

					num_ls = len(ls)		#第二个pny的同音异字字个数

					for k in range(0, num_ls):	#遍历第二个候选同音异字字
						#tuple_word = ['',0.0]
						tuple_word = list(list_words[j]) # 把现有的第一个候选字取出来
						#print('tw1: ',tuple_word)
						tuple_word[0] = tuple_word[0] + ls[k] # 将所有第一个候选字和第二个同音异字字组合
						#print('ls[k]  ',ls[k])

						'''取最后两个字体现1-gram
					    此处tuple_word[0]存储的是当前短语和当前同音异字待定组合'''
						tmp_words = tuple_word[0][-2:] #倒着取，一次在list中取2个字(1-gram)


						#print('tmp_words: ',tmp_words,tmp_words in self.model2)

						if(tmp_words in self.model2): # 判断它们是不是在状态转移表里(language_model2.txt)
							#print(tmp_words,tmp_words in self.model2)

							'''1-gram核心！在当前概率上乘转移概率，公式化简后为第n-1和n个字出现的次数除以第n-1个字出现的次数
							此处tuple_word[1]存储的概率是当前2个候选汉字的联合概率'''
							tuple_word[1] = tuple_word[1] * float(self.model2[tmp_words]) / float(self.model1[tmp_words[-2]])

							#print(self.model2[tmp_words],self.model1[tmp_words[-2]])
						else:
							tuple_word[1] = 0.0		#如果没有出现在状态转移表中，为超低频词，概率设为0，之后会丢弃该候选字
							continue
						#print('tw2: ',tuple_word)
						#print(tuple_word[1] >= pow(yuzhi, i))
						if(tuple_word[1] >= pow(yuzhi, i)):  #pow(x,y) = x**y   yuzhi=阈值
							# 大于阈值之后保留，否则丢弃

							'''list_word_2是大于阈值的两字词的临时存储列表'''
							list_words_2.append(tuple_word)

				'''遍历完所有二字组合后留下的概率超过阈值的交付list_words存储'''
				list_words = list_words_2
				#print(list_words,'\n')
		#print(list_words)

		#冒泡排序 递减
		for i in range(0, len(list_words)):
			for j in range(i + 1, len(list_words)):
				if(list_words[i][1] < list_words[j][1]):
					tmp = list_words[i]
					list_words[i] = list_words[j]
					list_words[j] = tmp
		
		return list_words  #存储所有候选汉字结果的列表，概率递减
		pass
		
	def GetSymbolDict(self, dictfilename):
		'''
		读取拼音汉字的字典文件
		返回读取后的字典
		'''
		txt_obj = open(dictfilename, 'r', encoding='UTF-8') # 打开文件并读入
		txt_text = txt_obj.read()
		txt_obj.close()
		txt_lines = txt_text.split('\n') # 文本分割
		
		dic_symbol = {} # 初始化符号字典
		for i in txt_lines:
			list_symbol=[] # 初始化符号列表
			if(i!=''):
				txt_l=i.split('\t')
				pinyin = txt_l[0]
				for word in txt_l[1]:
					list_symbol.append(word)
			dic_symbol[pinyin] = list_symbol
		
		return dic_symbol   #pny2word
		
	def GetLanguageModel(self, modelLanFilename):
		'''
		读取语言模型的文件（词频统计文件）
		返回读取后的模型
		'''
		txt_obj = open(modelLanFilename, 'r', encoding='UTF-8') # 打开文件并读入
		txt_text = txt_obj.read()
		txt_obj.close()
		txt_lines = txt_text.split('\n') # 文本分割
		
		dic_model = {} # 初始化符号字典
		for i in txt_lines:
			if(i!=''):
				txt_l=i.split('\t')
				if(len(txt_l) == 1):
					continue
				#print(txt_l)
				dic_model[txt_l[0]] = txt_l[1]  #{ 词 : 词频 ，……}
				
		return dic_model   #模型字典
	
	def GetPinyin(self, filename):
		file_obj = open(filename,'r',encoding='UTF-8')
		txt_all = file_obj.read()
		file_obj.close()
	
		txt_lines = txt_all.split('\n')
		dic={}
	
		for line in txt_lines:
			if(line == ''):
				continue
			pinyin_split = line.split('\t')
			
			list_pinyin=pinyin_split[0]
			
			if(list_pinyin not in dic and int(pinyin_split[1]) > 1): #构建词频大于1的pny字典  { pny : 词频 ，……}
				dic[list_pinyin] = pinyin_split[1]
		return dic


if(__name__=='__2main__'):
	
	ml = ModelLanguage('model_language')
	ml.LoadModel()
	
	#str_pinyin = 'jin1 tian1 shi4 san1 yue4 de5 yi1 tian1'
	#str = 'wo3 zhun3 bei4 chu1 fa1 le5'
	str = ''

	#str = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']

	pinyin = str.split(' ')

	#str_pinyin = ['da4','jia1','hao3','wo3','shi4','zhao4','hao4','ran2']

	r=ml.SpeechToText(pinyin)
	print('语音转文字结果：\n',r)

if (__name__ == '__main__'):
	ml = ModelLanguage('model_language')
	ml.LoadModel()

	# str_pinyin = ['zhe4','zhen1','shi4','ji2', 'hao3','de5']
	# str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
	# str_pinyin = ['ni3', 'hao3','a1']
	# str_pinyin = ['wo3','dui4','shi4','mei2','cuo4','ni3','hao3']

	# str_pinyin = ['da4','jia1','hao3','wo3','shi4','zhao4','hao4','ran2']

	# str_pinyin = ['wo3','dui4','shi4','tian1','mei2','na5','li3','hai4']
	# str_pinyin = ['ba3','zhe4','xie1','zuo4','wan2','wo3','jiu4','qu4','shui4','jiao4']
	# str_pinyin = ['wo3','qu4','a4','mei2','shi4','er2','la1']
	# str_pinyin = ['wo3', 'men5', 'qun2', 'li3', 'xiong1', 'di4', 'jian4', 'mei4', 'dou1', 'zai4', 'shuo1']
	# str_pinyin = ['su1', 'an1', 'ni3', 'sui4', 'li4', 'yun4', 'sui2', 'cong2', 'jiao4', 'ming2', 'tao2', 'qi3', 'yu2', 'peng2', 'ya4', 'yang4', 'chao1', 'dao3', 'jiang1', 'li3', 'yuan2', 'kang1', 'zhua1', 'zou3']
	# str_pinyin = ['da4', 'jia1', 'hao3']
	str_pinyin = ['kao3', 'yan2', 'yan1', 'yu3', 'ci2', 'hui4']
	# r = ml.decode(str_pinyin)
	r = ml.SpeechToText(str_pinyin)
	print('语音转文字结果：\n', r)