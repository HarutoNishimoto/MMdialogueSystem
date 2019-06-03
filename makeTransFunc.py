# -*-coding: utf-8 -*-

### 信念の更新式をデータから計算して出します．
### 計算した結果をテキストに出力します
### ここの関数はすべて1回しか使用しません．

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
		
	# 更新（あるなら更新，ないならそのまま）
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


# 状態遷移確率をデータから決定
# これはaction入りのもの
def updateSTparamsWithAction(df, f_param):
	# read
	state = df['label'].values
	state = ['o' if x > 0.5 else 'x' for x in state]
	df['label'] = state
	state_type = ['o', 'x']
	da = df['dialogue_act'].values.tolist()
	# da8は多いのでまとめる
	for i, val in enumerate(da):
		if val in ['oa', 'na', 'pa', 'op']:
			da[i] = 'down'
		else:
			da[i] = 'else'
	df['dialogue_act'] = da
	da = list(set(da))
	print(da)
	# read
	userID = pd.read_csv('./refData/userID.txt', header=None)
	userID = userID.iloc[:, 0].values
	# make dict
	allDict = {}
	for ele in da:
		allDict[ele] = {}

	for key in allDict.keys():
		for trans in itertools.product(state_type, repeat=2):
			allDict[key]['->'.join(list(trans))] = 0
	# count
	for ID in userID:
		df_cnt = df[df['name'].str.startswith(ID)]
		state = df_cnt['label'].values
		da = df_cnt['dialogue_act'].values
		for i, (s, d) in enumerate(zip(state[:-1], da[:-1])):
			trans = state[i] + '->' + state[i+1]
			allDict[da[i+1]][trans] = allDict[da[i+1]][trans] + 1

	# 和が1になるように変換
	for key in allDict.keys():
		for state in state_type:
			sum = 0.0
			for k in allDict[key].keys():
				if k.split('->')[0] == state:
					sum += allDict[key][k]
			for k in allDict[key].keys():
				if k.split('->')[0] == state:
					allDict[key][k] = float(allDict[key][k])/sum
		
	# 更新（あるなら更新，ないならそのまま）
	with open(f_param, 'r')as rf:
		para_data = rf.readlines()
	for key in allDict.keys():
		for k in allDict[key].keys():
			for i, val in enumerate(para_data):
				# 'op_o->x'みたいに記述します．
				if val.startswith('{}_{}'.format(key, k)):
					para_data[i] = para_data[i].split('=')
					para_data[i][1] = str(round(allDict[key][k], 3)) + '\n'
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


############################# 効果なし ################################

# 状態遷移確率をデータから決定
def updateSTparamsRemEven(f_csv, f_param):
	# read
	df = pd.read_csv(f_csv)
	state = df['label'].values

	state_bin = [''] * len(state)
	for i, val in enumerate(state):
		if val > 0.5:
			state_bin[i] = 'o'
		elif val < 0.5:
			state_bin[i] = 'x'
		else:
			state_bin[i] = 't'

	df['label'] = state_bin
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
			if 't' not in trans:
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

	# 更新（あるなら更新，ないならそのまま）
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




if __name__ == '__main__':
	import offline
	# comb
	df = pd.read_csv('190527_sysutte_da.csv')
	df = df.rename(columns={'userID':'name'})
	df = df.drop('dialogue_act', axis=1)
	df = df.rename(columns={'190527_da':'dialogue_act'})
	df = offline.remNOUSE(df)

	updateSTparamsWithAction(df, 'parameters.txt')



