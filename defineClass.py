# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
import MeCab

class turnIdx:
	def __init__(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def cntup(self, action):
		if action == "hold":
			self.idx_hold += 1
		elif action == "dig":
			self.idx_dig += 1
		elif action == "change":
			self.idx_change += 1

	def reset(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def continuity(self):
		self.cont_num += 1

class userImpression:
	def __init__(self, queue_size=3):
		self.prev_UI = [''] * queue_size
		self.current_UI = 0

	def update(self, UI):
		self.prev_UI.append(self.current_UI)
		del self.prev_UI[0]
		self.current_UI = UI

	def getUserImpression(self):
		return self.current_UI

	def show(self):
		print('prev', self.prev_UI)
		print('current', self.current_UI)




class params:
	def __init__(self):
		self.params = {}
		self.param_file_name = './../refData/parameters.txt'
		self.cls_num = 20

		with open(self.param_file_name, 'r')as f:
			paramInfo = f.readlines()
		paramInfo = [x.replace('\n', '') for x in paramInfo]
		paramInfo = [x for x in paramInfo if x != '']
		paramInfo = [x for x in paramInfo if '#' not in x]
		paramInfo = [x.split('=') for x in paramInfo]
		for i, val in enumerate(paramInfo):
			self.params[paramInfo[i][0]] = paramInfo[i][1]

	def set(self, param_name, param_value):
		self.params[param_name] = param_value

	def get(self, param_name, system_action_type=''):

		if (param_name == 'priprob_UI3') or (param_name == 'STP_UI3'):
			# クラス名を読み込み，それを数字に変換
			ACTindex = {}
			df = pd.read_csv(self.params['path_main_class_info'])
			act = sorted(list(set(df['cls'].values)))
			for i, val in enumerate(act):
				ACTindex[val] = i

		if param_name == 'priprob_UI3':
			priprob = np.load(self.params['path_priprob'])
			priprob = priprob[:, ACTindex[system_action_type]]
			for i, val in enumerate(priprob):
				ratio = lambda x:x/sum(x)
				if int(val) == 0:
					priprob[i] = 1	# 要素が0なら1で埋める
			return ratio(priprob)
		elif param_name == 'STP_UI3':
			STPall = np.load(self.params['path_STP'])
			STP = STPall[:, :, ACTindex[system_action_type]]
			STP = np.where(STP == 0, 1, STP)	# 要素が0なら1で埋める
			return STP
		else:
			return self.params[param_name]

	def write(self):
		with open(self.param_file_name, 'r')as rf:
			paramInfo = rf.readlines()

		for key, val in self.params.items():
			existing = False
			for i, line in enumerate(paramInfo):
				if line.startswith(key):
					paramInfo[i] = key + '=' + str(val) + '\n'
					existing = True
					break
			if not existing:
				paramInfo.append(key + '=' + str(val) + '\n')

		with open(self.param_file_name, 'w')as wf:
			for val in paramInfo:
				wf.write(val)

	def show(self):
		for key, val in self.params.items():
			print(key, val)



class historySysUtte:
	def __init__(self):
		self.history_sysutte = []
		self.history_sysutte_default = {}
		self.history_sysutte_class = []

	def add_sysutte(self, utterance, theme):
		if theme == 'default':
			if utterance in self.history_sysutte_default:
				self.history_sysutte_default[utterance] += 1
			else:
				self.history_sysutte_default[utterance] = 1
		else:
			self.history_sysutte.append(utterance)

	def add_sysutte_class(self, clas):
		self.history_sysutte_class.append(clas)

	def get_prev_sysutte_class(self):
		print(self.history_sysutte_class)
		return self.history_sysutte_class[-1]

class historyUserWord:
	def __init__(self):
		self.history_user_word = {}
		self.user_latest_new_word = []

	def add(self, utterance):
		if utterance == None:
			pass
		else:
			self.user_latest_new_word = []
			mt = MeCab.Tagger()
			node = mt.parseToNode(utterance)
			while node:
				fields = node.feature.split(",")
				if fields[0] == '名詞' and len(fields) > 7:
					word = fields[6]
					if word in self.history_user_word:
						self.history_user_word[word] += 1
					else:
						self.history_user_word[word] = 1
						self.user_latest_new_word.append(word)
				node = node.next

	def show(self):
		for key, val in self.history_user_word.items():
			print(val, key)

	def getNewWord(self):
		return self.user_latest_new_word




class historyTheme:
	def __init__(self, random_choice=True):
		self.allTheme = list(pd.read_csv('./../refData/usingTheme.txt', header=None)[0].values)
		self.random_choice = random_choice
		if self.random_choice == True:
			self.nowTheme = random.choice(self.allTheme)
		else:
			print('使用する話題をindexで指定してください')
			for i, val in enumerate(self.allTheme):
				print(i, val)
			index = int(input('>> '))
			self.nowTheme = self.allTheme[index]
		self.historyTheme = []
		self.nowTheme_ExchgNum = 1
		self.history_UI3 = []
		self.max_exchange_num_1theme = 10
		self.min_exchange_num_1theme = 5
		self.low_UI3_border = 4.0

		self.allTheme.remove(self.nowTheme)

	def decideNextTheme(self, UI):
		# 変更可否
		if self.nowTheme_ExchgNum > self.max_exchange_num_1theme:
			chg = True
		elif (self.nowTheme_ExchgNum > self.min_exchange_num_1theme) and (np.mean(self.history_UI3) < self.low_UI3_border):
			chg = True
		else:
			chg = False
		# 変更
		if chg:
			self.historyTheme.append(self.nowTheme)
			if self.random_choice == True:
				self.nowTheme = random.choice(self.allTheme)
			else:
				print('使用する話題をindexで指定してください')
				for i, val in enumerate(self.allTheme):
					print(i, val)
				index = int(input('>> '))
				self.nowTheme = self.allTheme[index]
			self.allTheme.remove(self.nowTheme)
			self.nowTheme_ExchgNum = 1
			self.history_UI3 = []
		else:
			# 1発話の場合
			if (len(self.historyTheme) == 0) and (len(self.history_UI3) == 0):
				self.nowTheme_ExchgNum = 1
				self.history_UI3.append(UI)
			else:
				self.nowTheme_ExchgNum += 1
				self.history_UI3.append(UI)		

		return self.nowTheme

	def show(self):
		if len(self.history_UI3) > 0:
			print(self.nowTheme[0], self.nowTheme_ExchgNum, np.mean(self.history_UI3))
		else:
			print(self.nowTheme[0], self.nowTheme_ExchgNum, '')






if __name__ == '__main__':


	pass


