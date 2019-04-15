# -*- coding:utf-8 -*-

# ここはFSTにコマンドメッセージを送るモジュール
import math
import numpy as np
import main
import pandas as pd

import matplotlib.pyplot as plt


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


# ソフトマックス関数
def logit(a):
	x = np.arange(0.0, 1.01, 0.01)
	y = (1/a) * np.log(x/(1-x))

	plt.plot(x, y)
	plt.show()

	

# 興味ありなし（連続値）を引数にしてコマンドメッセージを作成
def makeCommandMessage(action, state, interest, theme, turnIdx):
	thres1, thres2 = getParams('policy')
	
	# 保留(hold)、掘り下げ(dig)、切り替え(change)とする
	if thres2 <= state[0][0]:
		send_message = "RECOG_EVENT_STOP|{}|dig|{}".format(theme, turnIdx.idx_dig)
		action = "dig"
		turnIdx.cntup("dig")
	elif (thres1 <= state[0][0]) and (state[0][0] < thres2):
		send_message = "RECOG_EVENT_STOP|{}|hold|{}".format(theme, turnIdx.idx_hold)
		action = "hold"
		turnIdx.cntup("hold")
	else:
		send_message = "RECOG_EVENT_STOP|{}|change|{}".format(theme, turnIdx.idx_change)
		action = "change"
	print(send_message)
	send_message = send_message.encode("shift_jis")

	return send_message, state, action, turnIdx


# 政策関数
def policy(s_1, thres=None):
	if thres == None:
		thres1, thres2 = getParams('policy')
	else:
		thres1, thres2 = thres[0], thres[1]
	# 保留(hold)、掘り下げ(dig)、切り替え(change)とする
	if thres2 <= s_1:
		action = "dig"
	elif (thres1 <= s_1) and (s_1 < thres2):
		action = "hold"
	else:
		action = "change"

	return action

# 信念の更新
# 現在の興味と累積のstateを引数にして、次時刻のstateを計算
def updateStatePOMDP(state, interest, flg):
	# 現在の興味度
	inte = interest.get_interest()	# LR
	# 新提案手法（inteに別の値を入れる）
	#if (inte < 0.2) or (0.6 < inte):
	#	inte = interest.get_prev_ave_interest()

	# 新提案尤度
	prob = softmax([inte, 1-inte])

	# 尤度
	#prob = np.array([inte, 1 - inte])
	priprob = getParams('priprob')
	likelihood = np.diag(prob / priprob)
	# 状態遷移確率
	STP = getParams('STP')
	#print(likelihood)
	# 計算
	if flg:
		next_state = likelihood @ [[0.5], [0.5]]
	else:
		next_state = likelihood @ STP @ state
	# 正規化係数
	next_state = next_state/np.sum(next_state, axis=0)[0]

	return next_state


# 外部指定のパラメータを読み込み
def getParams(return_type, filename='parameters.txt'):
	with open(filename, 'r')as f:
		paramInfo = f.readlines()
	paramInfo = [x.replace('\n', '') for x in paramInfo]
	paramInfo = [x for x in paramInfo if x != '']
	paramInfo = [x for x in paramInfo if '#' not in x]
	paramInfo = [x.split('=') for x in paramInfo]

	paramDict = {}
	for i, val in enumerate(paramInfo):
		paramDict[paramInfo[i][0]] = float(paramInfo[i][1])

	if return_type == 'STP':
		STP = np.array([[paramDict['o->o'], paramDict['x->o']],
			[paramDict['o->x'], paramDict['x->x']]])
		return STP
	elif return_type == 'policy':
		return paramDict['p_thres1'], paramDict['p_thres2']
	elif return_type == 'topic_continue':
		return paramDict['t_thres1'], paramDict['t_thres2']
	elif return_type == 'priprob':
		return [paramDict['prob_o'], paramDict['prob_x']]
	elif return_type == 'sigmoid':
		return paramDict['probA'], paramDict['probB']
	if return_type == 'score':
		score = np.array([[paramDict['d->d'], paramDict['d->h'], paramDict['d->c']],
		[paramDict['h->d'], paramDict['h->h'], paramDict['h->c']],
		[paramDict['c->d'], paramDict['c->h'], paramDict['c->c']]])
		return score
	else:
		print('invalid return type')
		
	return None


# 0-1の連続値に変換（0（興味なし）から1（興味あり））
def sigmoid(x, print=False):
	probA, probB = getParams('sigmoid')
	exp = math.e
	s = 1.0 / (1 + exp**(x * probA + probB))

	if print:
		import matplotlib.pyplot as plt
		x, y = np.arange(-5.0, 5.0, 0.05), []
		for num in x:
			y.append( 1.0 / (1 + exp**(num * probA + probB)))
		plt.plot(x, y)
		plt.show()

	return s

	

############ 使用していない ##################


# 前回のinterestと今回のinterestを比べただけの簡単な報酬
def calReward(reward, inte):
	if (inte.cnt_interest >= 0) and (inte.prev_interest >= 0):
		reward = 5
	if (inte.cnt_interest >= 0) and (inte.prev_interest < 0):
		reward = 20
	if (inte.cnt_interest < 0) and (inte.prev_interest >= 0):
		reward = -10
	if (inte.cnt_interest < 0) and (inte.prev_interest < 0):
		reward = -1
	
	return reward


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

############ 使用していない ##################



if __name__ == "__main__":

	tex = 'g;lfdsndc;lk{}{}'




	



