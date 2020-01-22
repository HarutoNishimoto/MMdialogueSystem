# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import MeCab



# パラメータ持ちです
class params:
	def __init__(self):
		self.params = {}
		self.param_file_name = '/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/refData/parameters.txt'

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


# themeを管理してます
class historyTheme:
	def __init__(self, random_choice=True):
		p = params()
		self.allTheme = list(pd.read_csv(p.get('path_using_theme'), header=None)[0].values)
		self.random_choice = random_choice
		self.nowTheme_ExchgNum = 0
		self.history_impression_1theme = []
		self.max_exchange_num_1theme = 10
		self.min_exchange_num_1theme = 5
		self.low_UI3_border = 3

	# 最初，話題変更する時はUIにNoneを入れること
	def decideNextTheme(self, UI):
		# 変更可否の決定
		if UI == None:
			chg = True
		elif self.nowTheme_ExchgNum >= self.max_exchange_num_1theme-1:
			chg = True
		elif (self.nowTheme_ExchgNum >= self.min_exchange_num_1theme-1) and \
		 (np.mean(self.history_impression_1theme) < self.low_UI3_border):
			chg = True
		else:
			chg = False

		if chg:# 変更
			if self.random_choice:
				self.nowTheme = np.random.choice(self.allTheme)
			else:
				print('使用する話題をindexで指定してください')
				for i, val in enumerate(self.allTheme):
					print(i, val)
				index = int(input('>> '))
				self.nowTheme = self.allTheme[index]
			self.allTheme.remove(self.nowTheme)
			self.nowTheme_ExchgNum = 0
			self.history_impression_1theme = []
		else:# 変更しない
			self.nowTheme_ExchgNum += 1
			self.history_impression_1theme.append(UI)

		return chg, self.nowTheme



# システム発話の管理します
class historySysUtte:
	def __init__(self):
		self.history_sysutte = []
		self.history_sysutte_class = []

	def add_sysutte(self, utterance, clas):
		self.history_sysutte.append(utterance)
		self.history_sysutte_class.append(clas)




# ユーザの重要そうな単語だけを保持します
# 今は使用していません（今後いるかもしれません）
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


if __name__ == '__main__':
	pass


