# -*- coding:utf-8 -*-

# ここはFSTにコマンドメッセージを送るモジュール
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import main
import random


# ソフトマックス関数
# coefは推定値の振れ幅を調整するためのもの．（デフォルトは1）
def softmax(a, coef=1):
	# 一番大きい値を取得
	c = np.max(a)
	# 各要素から一番大きな値を引く（オーバーフロー対策）
	exp_a = np.exp(coef * (a - c))
	sum_exp_a = np.sum(exp_a)
	# 要素の値/全体の要素の合計
	y = exp_a / sum_exp_a
	return y 


# 現在の興味と累積のstateを引数にして、次時刻のstateを計算
def updateStatePOMDP(state, interest, da):
	inte = interest.get_interest()
	# 尤度
	## 推定値
	### 正規分布の定義
	myu = inte
	sigma = getParams('sigma')
	ND = lambda x: (math.exp(-(x-myu)**2/(2*sigma**2))) / math.sqrt(2*math.pi)
	prob = np.array([ND(1), ND(2), ND(3), ND(4), ND(5), ND(6), ND(7)])
	## 事前確率
	priprob = getParams('priprob_UI3', da)
	alpha = getParams('alpha')
	priprob = np.power(priprob, alpha)
	likelihood = np.diag(prob / priprob)
	# 状態遷移確率
	STP = getParams('STP_UI3', da)
	# 更新
	next_state = likelihood @ STP @ state
	next_state = next_state/np.sum(next_state, axis=0)[0]
	return next_state


# パラメータの書き換え
def setParam(param_name, param_value, param_file='/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem-master/parameters.txt'):

	with open(param_file, 'r')as rf:
		paramInfo = rf.readlines()

	# set values
	for i, val in enumerate(paramInfo):
		if val.startswith(param_name):
			paramInfo[i] = param_name + '=' + str(param_value) + '\n'

	with open(param_file, 'w')as wf:
		for val in paramInfo:
			wf.write(val)



# 外部指定のパラメータを読み込み
def getParams(return_type, system_action_type=''):
	# ファイル名指定
	#param_main_file, param_STP_file, param_pprob_file = 'parameters.txt', 'param_STP_UI3_withA.npy', 'param_priprob_UI3_withA.npy'
	param_main_file, param_STP_file, param_pprob_file = 'parameters.txt', 'takedaT_STP.npy', 'param_priprob_UI3_withA.npy'

	with open(param_main_file, 'r')as f:
		paramInfo = f.readlines()
	paramInfo = [x.replace('\n', '') for x in paramInfo]
	paramInfo = [x for x in paramInfo if x != '']
	paramInfo = [x for x in paramInfo if '#' not in x]
	paramInfo = [x.split('=') for x in paramInfo]
	PD = {}
	for i, val in enumerate(paramInfo):
		PD[paramInfo[i][0]] = float(paramInfo[i][1])

	# da index
	DAindex = {}
	da = ['io','na','no','oa','op','pa','qw','qy']
	for i, val in enumerate(da):
		DAindex[val] = i

	if return_type == 'STP':
		if system_action_type == '':
			da = ''
		else:
			da = system_action_type + '_'
		STP = np.array([[PD[da + 'o->o'], PD[da + 'x->o']],
			[PD[da + 'o->x'], PD[da + 'x->x']]])
		return STP
	elif return_type == 'priprob':
		return [PD['prob_o'], PD['prob_x']]
	elif return_type == 'sigmoid':
		return PD['probA'], PD['probB']
	elif return_type == 'priprob_UI3':
		priprob = np.load(param_pprob_file)
		priprob = priprob[:, DAindex[system_action_type]]
		for i, val in enumerate(priprob):
			ratio = lambda x:x/sum(x)
			if int(val) == 0:
				priprob[i] = 1	# 要素が0なら1で下駄を履かす
		return ratio(priprob)
	elif return_type == 'STP_UI3':
		STP = np.load(param_STP_file)
		return STP[:, :, DAindex[system_action_type]]
	elif return_type == 'alpha':
		return PD['alpha']
	elif return_type == 'sigma':
		return PD['sigma']
	else:
		print('invalid return type')		
	return None


	

############ 使用していない ##################

# 報酬をもとに、STPを更新
def updateSTP(action, reward):
	df = pd.read_csv("STP_params.csv")
	meta = ["_s11", "_s21", "_s12", "_s22"]
	meta = [action + x for x in meta]
	cnt_STP = df[meta].values[-1]
	prev_STP = df[meta].values[-2]
	# 更新
	alpha = 0.2
	gamma = 0.99
	ret_STP = (1 - alpha) * cnt_STP + alpha * (reward + gamma * prev_STP)
	# ファイルに追加
	add = df.iloc[-1, :]
	add[meta] = ret_STP
	add_STP = ",".join([str(x) for x in add.values.tolist()])
	with open("STP_params.csv", "a")as f:
		f.write(add_STP + "\n")
	return ret_STP.reshape([2, 2])

# 政策関数
def Policy(belief, current_theme):

	action_class_file = '/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/sysutte_fea8_EA_KM_stand.csv'
	theme_file = '/Users/haruto/Desktop/mainwork/codes/1902themeInfo/1902MMcorpus_theme_edit.csv'

	ACTdf = pd.read_csv(action_class_file)
	THEMEdf = pd.read_csv(theme_file)
	df = pd.merge(ACTdf, THEMEdf, on='agent_utterance')
	df = df[['cls','theme','agent_utterance']]

	############ ここをメインで変更する（強化学習により）
	# 適当に設計しています
	if belief[0, 0] > 0.6 and belief[0, 0] > 0.5:
		next_action_cls = random.choice([0,1,5])
	elif belief[0, 0] > 0.6 and belief[0, 0] < 0.5:
		next_action_cls = random.choice([2,3,4])
	else:
		next_action_cls = random.choice([6,7,8,9])
	############ ここをメインで変更する（強化学習により）



	CANDIDATEdf = df[(df['cls'] == next_action_cls) & ((df['theme'] == current_theme) | (df['theme'] == 'default'))]
	print(CANDIDATEdf)

	next_sysUtte_candidate = CANDIDATEdf['agent_utterance'].values
	# 候補からランダム選択
	next_sysUtte = random.choice(next_sysUtte_candidate)
	return next_sysUtte


if __name__ == "__main__":

	belief = np.array([[0.7, 0.3]]).transpose()
	print('入力の信念（状態である確率）')
	print(belief)
	next_sysUtte = Policy(belief, 'スポーツ')
	print('選択されたシステム発話')
	print(next_sysUtte)




	df = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/dialogue.csv')
	plt.hist(df['reaction'].values, bins=30)
	plt.show()
