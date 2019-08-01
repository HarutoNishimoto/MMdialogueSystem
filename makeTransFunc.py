# -*-coding: utf-8 -*-

### 信念の更新式をデータから計算して出します．
### 計算した結果をテキストに出力します
### ここの関数はすべて1回しか使用しません．

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

#################### ここはInteの時のPOMDP ####################################

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

#################### ここはInteの時のPOMDP ####################################


# 状態遷移確率をデータから決定
def updateSTparams_UI3_withA(f_label, f_da, f_param):
	UIdf = pd.read_csv(f_label)
	state = UIdf['label'].values
	state = [int(round(x)) for x in state]
	UIdf['label'] = state
	UIdf = UIdf.rename(columns={'name': 'userID'})
	UIdf = UIdf[['userID', 'label']]

	DAdf = pd.read_csv(f_da)
	DAdf = DAdf[['userID', 'dialogue_act']]
	df = pd.merge(UIdf, DAdf, on='userID')
	# userID
	userID = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID = userID.iloc[:, 0].values
	# da
	DAindex = {}
	da = list(set(df['dialogue_act'].values))
	da = np.sort(da)
	
	for i, val in enumerate(da):
		DAindex[val] = i
	# main mat
	freq_matrix = np.zeros((7, 7, 8))

	for ID in userID:
		df_cnt = df[df['userID'].str.startswith(ID)]
		for i, (ui, da) in enumerate(zip(df_cnt['label'].values, df_cnt['dialogue_act'].values)):
			if i == 0:
				freq_matrix[3][ui-1][DAindex[da]] += 1	# 4からの遷移とする
			else:
				freq_matrix[df_cnt['label'].values[i-1]-1][ui-1][DAindex[da]] += 1
	# matrix(7 * 7 * 8)
	# 7:state 7:next_state 8:dialogue_act
	np.save(f_param, freq_matrix)



# 状態の事前確率をデータから決定
def updatePriProb_UI3_withA(f_label, f_da, f_param):
	UIdf = pd.read_csv(f_label)
	state = UIdf['label'].values
	state = [int(round(x)) for x in state]
	UIdf['label'] = state
	UIdf = UIdf.rename(columns={'name': 'userID'})
	UIdf = UIdf[['userID', 'label']]

	DAdf = pd.read_csv(f_da)
	DAdf = DAdf[['userID', 'dialogue_act']]
	df = pd.merge(UIdf, DAdf, on='userID')

	freq_matrix = np.zeros((7, 8))
	DAindex = {}
	da = list(set(df['dialogue_act'].values))
	da = np.sort(da)
	for i, val in enumerate(da):
		DAindex[val] = i

	for ui, da in zip(df['label'].values, df['dialogue_act'].values):
		freq_matrix[ui-1][DAindex[da]] += 1

	# matrix(7 * 8)
	# 7:state 8:dialogue_act
	np.save(f_param, freq_matrix)

# 武田先生用に作成したもの
def makeSTPInfoCsv():
	f_label, f_da = './../pred_UI3_svr.csv', './../UI3_su_da.csv'

	UIdf = pd.read_csv(f_label)
	state = UIdf['label'].values
	UIdf = UIdf.rename(columns={'name': 'userID'})
	UIdf = UIdf[['userID', 'label']]

	DAdf = pd.read_csv(f_da)
	DAdf = DAdf[['userID', 'dialogue_act']]
	df = pd.merge(UIdf, DAdf, on='userID')
	# userID
	userID = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID = userID.iloc[:, 0].values
	# da
	DAindex = {}
	da = list(set(df['dialogue_act'].values))
	da = np.sort(da)
	
	for i, val in enumerate(da):
		DAindex[val] = i

	new_df = pd.DataFrame(data=[['a','a','a','a']], columns=['index', 'state_i-1', 'action', 'state_i'])
	for ID in userID:
		df_cnt = df[df['userID'].str.startswith(ID)]
		for i, (index, ui, da) in enumerate(zip(df_cnt['userID'].values, df_cnt['label'].values, df_cnt['dialogue_act'].values)):
			if i == 0:
				s = pd.Series([index, 4.0, ui, DAindex[da]], index=new_df.columns, name=index)
				new_df = new_df.append(s)
			else:
				s = pd.Series([index, df_cnt['label'].values[i-1], ui, DAindex[da]], index=new_df.columns, name=index)
				new_df = new_df.append(s)
	new_df.to_csv('UI3_trans.csv', index=None)

# 武田先生のSTP読み込み
def readSTPCsv():
	df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem-master/trans_all.csv')
	tensor = np.zeros((7, 7, 8))
	for index, row in df.iterrows():
		tensor[int(row['state_i-1']-1)][int(row['state_i']-1)][int(row['action'])] = row['prob']
	np.save('takedaT_STP.npy', tensor)


if __name__ == '__main__':

	#updatePriProb_UI3_withA('./../pred_UI3_svr.csv', './../UI3_su_da.csv', 'priprob_UI3_withA.npy')

	#updateSTparams_UI3_withA('./../pred_UI3_svr.csv', './../UI3_su_da.csv', 'STP_UI3_withA.npy')


	#readSTPCsv()






