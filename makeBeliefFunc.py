# -*-coding: utf-8 -*-

### 信念の更新式をデータから計算して出します．
### 計算した結果をテキストに出力します
### ここの関数はすべて1回しか使用しません．

import pandas as pd
import numpy as np
import defineClass


# 状態の事前確率をデータから決定，配列で保存
def getPriProb(f_state, f_action, f_param, action_num):
	UIdf = pd.read_csv(f_state)
	state = UIdf['label'].values
	state = [int(round(x)) for x in state]
	UIdf['int(state)'] = state
	ACTdf = pd.read_csv(f_action)
	df = pd.merge(UIdf, ACTdf, on='name')
	# act-class
	ACTindex = {}
	act = np.sort(list(set(df['cls'].values)))
	for i, val in enumerate(act):
		ACTindex[val] = i

	freq_matrix = np.zeros((7, action_num))
	for ui, a in zip(df['int(state)'].values, df['cls'].values):
		freq_matrix[ui-1][ACTindex[a]] += 1
	np.save(f_param, freq_matrix)
	print(freq_matrix)


# 状態遷移確率をデータから決定，配列で保存
def getSTP(f_state, f_action, f_param, action_num):
	UIdf = pd.read_csv(f_state)
	state = UIdf['label'].values
	state = [int(round(x)) for x in state]
	UIdf['int(state)'] = state
	ACTdf = pd.read_csv(f_action)
	df = pd.merge(UIdf, ACTdf, on='name')
	# userID
	userID = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID = userID.iloc[:, 0].values
	# act-class
	ACTindex = {}
	act = np.sort(list(set(df['cls'].values)))
	for i, val in enumerate(act):
		ACTindex[val] = i
	# main mat
	freq_matrix = np.zeros((7, 7, action_num))

	for ID in userID:
		df_cnt = df[df['name'].str.startswith(ID)]
		for i, (ui, a) in enumerate(zip(df_cnt['int(state)'].values, df_cnt['cls'].values)):
			if i == 0:
				freq_matrix[3][ui-1][ACTindex[a]] += 1	# 4からの遷移とする
			else:
				freq_matrix[df_cnt['int(state)'].values[i-1]-1][ui-1][ACTindex[a]] += 1
	np.save(f_param, freq_matrix)
	print(freq_matrix)


# 武田先生用にfile作成
def makeSTPInfoCsv():
	f_label, f_da = './../pred_UI3_svr.csv', '/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/UI3_su_da.csv'

	UIdf = pd.read_csv(f_label)
	state = UIdf['label'].values
	UIdf = UIdf[['name', 'label']]

	DAdf = pd.read_csv(f_da)
	DAdf = DAdf[['userID', 'dialogue_act']]
	DAdf = DAdf.rename(columns={'userID': 'name'})

	df = pd.merge(UIdf, DAdf, on='name')
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
		df_cnt = df[df['name'].str.startswith(ID)]
		for i, (index, ui, da) in enumerate(zip(df_cnt['name'].values, df_cnt['label'].values, df_cnt['dialogue_act'].values)):
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

	params = defineClass.params()


	# 事前確率の設計（カウント）
	getPriProb('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/UI3_su_da.csv',
		params.get('path_main_class_info'),
		params.get('path_priprob'),
		int(params.get('class_num')))

	# 状態遷移確率の設計（カウント）
	getSTP('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/UI3_su_da.csv',
		params.get('path_main_class_info'),
		params.get('path_STP'),
		int(params.get('class_num')))




