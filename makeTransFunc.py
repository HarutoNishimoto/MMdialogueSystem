# -*-coding: utf-8 -*-

### 信念の更新式をデータから計算して出します．
### 計算した結果をテキストに出力します

import pandas as pd
import itertools


# 状態遷移確率をデータから決定
def updateSTparams(f_csv, f_param):
	# read
	df = pd.read_csv(f_csv)
	state = df['label'].values
	state = ['o' if x > 0 else 'x' for x in state]
	df['label'] = state
	state_type = ['o', 'x']
	# read
	userID = pd.read_csv('./refData/userID.txt', header=None)
	userID = userID.iloc[:, 0].values
	# make dict
	Dict = {}
	for trans in itertools.product(state_type, repeat=2):
		key = '->'.join(list(trans))
		Dict[key] = 0
	# count
	for ID in userID:
		df_cnt = df[df['name'].str.startswith(ID)]
		state = df_cnt['label'].values
		for i, val in enumerate(state[:-1]):
			trans = state[i] + '->' + state[i+1]
			Dict[trans] = Dict[trans] + 1

	# 和が1になるように変換
	for state in state_type:
		sum = 0.0
		for key in Dict.keys():
			if key.split('->')[0] == state:
				sum += Dict[key]
		for key in Dict.keys():
			if key.split('->')[0] == state:
				Dict[key] = float(Dict[key])/sum
		
	# 更新
	with open(f_param, 'r')as rf:
		para_data = rf.readlines()
	for key in Dict.keys():
		for i, val in enumerate(para_data):
			if val.startswith(key):
				para_data[i] = para_data[i].split('=')
				para_data[i][1] = str(round(Dict[key], 3)) + '\n'
				para_data[i] = '='.join(para_data[i])
				break
	with open(f_param, 'w')as wf:
		for line in para_data:
			wf.write(line)

# 状態の事前確率をデータから決定
def updatePriProb(f_csv, f_param):
	import collections
	# read
	df = pd.read_csv(f_csv)
	state = df['label'].values
	state = ['prob_o' if x > 0 else 'prob_x' for x in state]
	Dict = collections.Counter(state)
	# 更新
	with open(f_param, 'r')as rf:
		para_data = rf.readlines()
	for key in Dict.keys():
		for i, val in enumerate(para_data):
			if val.startswith(key):
				para_data[i] = para_data[i].split('=')
				para_data[i][1] = str(round(float(Dict[key])/len(state), 3)) + '\n'
				para_data[i] = '='.join(para_data[i])
				break
	with open(f_param, 'w')as wf:
		for line in para_data:
			wf.write(line)

if __name__ == '__main__':

	#updateSTparams('fusion_all_reg.csv', 'parameters.txt')
	updatePriProb('fusion_all_reg.csv', 'parameters.txt')