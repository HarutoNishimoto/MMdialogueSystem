# -*- coding:utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import defineClass
import random
from tqdm import tqdm
import MeCab
from sklearn import preprocessing


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

# 興味(regression)を入力としてPOMとBASEの行動ラベルを作成
# これは収録データから出した推定UI3に対してPOMの影響を見るだけのもの
def getBeliefTrans_AllData(index, pred, true, action, params, csv=False):
	# make df
	df = pd.DataFrame(data=np.array([index, pred, true, action]).transpose(), columns=['name', 'pred', 'true', 'cls'])
	# init
	POMDP_state, label = [], []
	# userID
	userID = [x.split("_")[0] for x in df['name'].values]
	userID = np.sort(list(set(userID)))

	for ID in tqdm(userID, ascii=True):
		df_cnt = df[df['name'].str.startswith(ID)]
		pred_cnt, label_cnt, action_cnt = df_cnt['pred'].values, df_cnt['true'].values, df_cnt['cls'].values
		POMDP_state_cnt = [''] * len(pred_cnt)
		state = np.ones((7, 1))
		UI = defineClass.userImpression()
		next_state = state
		for i, (p, act) in enumerate(zip(pred_cnt, action_cnt)):
			UI.update(p)
			next_state = updateBelief(next_state, UI, act, params)
			exp = np.dot(next_state.flatten(), np.arange(1, len(next_state.flatten())+1, 1))
			POMDP_state_cnt[i] = exp
			label_cnt[i] = label_cnt[i]
		POMDP_state.extend(POMDP_state_cnt)
		label.extend(label_cnt)
	# BASE
	BASE_state = [''] * len(pred)
	for i, val in enumerate(pred):
		BASE_state[i] = val

	# write
	data = np.array([index, POMDP_state, BASE_state, label]).transpose()
	df = pd.DataFrame(data=data, columns=['name', 'POM_state', 'BASE_state', 'label'])
	if type(csv) is str:
		df.to_csv(csv, index=None)
	return df['POM_state'].values, df['BASE_state'].values





# 現在の興味と累積のstateを引数にして、次時刻のstateを計算
def updateBelief(state, UI, action, params):
	# 正規分布の定義
	current_UI = UI.getUserImpression()
	myu = current_UI
	sigma = float(params.get('sigma'))
	ND = lambda x: (math.exp(-(x-myu)**2/(2*sigma**2))) / math.sqrt(2*math.pi)
	# 話題変更のときだけ別の操作
	if action == 'change_theme':
		next_state = np.array([ND(x) for x in range(1, state.shape[0] + 1, 1)])
		next_state = next_state/np.sum(next_state)
		return next_state
	# 尤度
	## 推定値
	prob = np.array([ND(x) for x in range(1, state.shape[0] + 1, 1)])
	## 事前確率
	priprob = params.get('priprob_UI3', action)
	alpha = float(params.get('alpha'))
	priprob = np.power(priprob, alpha)
	likelihood = np.diag(prob / priprob)
	# 状態遷移確率
	STP = params.get('STP_UI3', action)
	# 更新
	next_state = likelihood @ STP @ state
	next_state = next_state/np.sum(next_state)
	return next_state


# mecabを用いて名詞の数と語長を返す
def getNounandLen(utterance):
	mt = MeCab.Tagger()
	node = mt.parseToNode(utterance)

	num_noun = 0
	utte_len = 0
	while node:
		fields = node.feature.split(",")
		if fields[0] != 'BOS/EOS':
			if fields[0] == '名詞':
				num_noun += 1
			if len(fields) > 7:
				utte_len += len(fields[7])
		node = node.next
	return num_noun, utte_len


# 政策関数
def Policy(belief, user_utterance, history_utterance, history_theme, model):

	# params init 
	params = defineClass.params()
	ACTdf = pd.read_csv(params.get('path_utterance_by_class'))
	THEMEdf = pd.read_csv(params.get('path_theme_info'))
	ACTdf = ACTdf.drop(['freq', 'theme'], axis=1)
	df = pd.merge(ACTdf, THEMEdf, on='agent_utterance')

	############ ここをメインで変更する（強化学習により）
	belief_1dim = np.dot(belief.flatten(), np.arange(1, len(belief.flatten())+1, 1))
	next_action_cls = 0
	### クラスの選択
	if model == 'euclid_dim3':
		candidate_cls_num = 3
		feadf = pd.read_csv(params.get('path_feature_average_by_class'))
		feadf['prevUI3'] = feadf['UI3average'].values - feadf['UI3average_diff'].values
		fea_ave = feadf[['prevUI3', 'num_noun_before', 'u_utte_len_before']].values

		if user_utterance != None:
			num_noun_cnt, u_utte_len_cnt = getNounandLen(user_utterance)
			fea_cnt = [[belief_1dim, num_noun_cnt, u_utte_len_cnt]]
			ss = preprocessing.StandardScaler()
			ss.fit(fea_ave)
			ave_ss = ss.transform(fea_ave)
			cnt_ss = ss.transform(fea_cnt)

			clas = np.sort(list(set(df['cls'].values)))
			distance = []
			for i, val in enumerate(clas):
				dist = np.linalg.norm(cnt_ss - ave_ss[i])
				distance.append(dist)

			Dict = dict(zip(clas, distance))
			candidate_cls = []
			for k, v in sorted(Dict.items(), key=lambda x: x[1])[:candidate_cls_num]:
				candidate_cls.append(k)
			print(candidate_cls)
			next_action_cls = np.random.choice(candidate_cls)
		else:
			next_action_cls = 0
	elif model == 'euclid_dim3_weighted':
		feadf = pd.read_csv(params.get('path_feature_average_by_class'))
		feadf['prevUI3'] = feadf['UI3average'].values - feadf['UI3average_diff'].values
		fea_ave = feadf[['prevUI3', 'num_noun_before', 'u_utte_len_before']].values

		if user_utterance != None:
			num_noun_cnt, u_utte_len_cnt = getNounandLen(user_utterance)
			fea_cnt = [[belief_1dim, num_noun_cnt, u_utte_len_cnt]]
			ss = preprocessing.StandardScaler()
			ss.fit(fea_ave)
			ave_ss = ss.transform(fea_ave)
			cnt_ss = ss.transform(fea_cnt)

			clas = np.sort(list(set(df['cls'].values)))
			distance = []
			for i, val in enumerate(clas):
				dist = np.linalg.norm(cnt_ss - ave_ss[i])
				distance.append(dist)
			distance = [1.0/x for x in distance]# 逆数
			distance = softmax(distance, coef=5)# 確率に
			print(distance)
			next_action_cls = np.random.choice(clas, p=distance)
		else:
			next_action_cls = 0
	elif model == 'hand_STP':
		if user_utterance != None:
			if history_utterance.get_prev_sysutte_class() == 'change_theme':
				next_action_cls = random.choice([0, 3])# 情報提供か，依存なし質問か
			else:
				STPdf = pd.read_csv(params.get('path_hand_STP'))
				STP_int = STPdf.iloc[:, :].values
				STP_prob = []
				for i, val in enumerate(STP_int):
					STP_prob.append(softmax(val, coef=2))
				STP_prob = np.array(STP_prob)
				next_action_cls = np.random.choice([0,1,2,3,4], p=STP_prob[history_utterance.get_prev_sysutte_class()].tolist())
		else:
			next_action_cls = 0
	elif model == 'base':
		pass

	### 発話の選択
	if (model == 'euclid_dim1') or (model == 'euclid_dim3') or (model == 'euclid_dim3_weighted'):
		CANDIDATEdf = df[(df['cls'] == next_action_cls) &
			((df['theme'] == history_theme.nowTheme) | (df['theme'] == 'default'))]
	elif model == 'base':
		CANDIDATEdf = df[(df['theme'] == history_theme.nowTheme) | (df['theme'] == 'default')]

	correct = False
	while not correct:
		# 話題変更した直後は専用の発話を使用しましょう．
		if history_theme.nowTheme_ExchgNum == 1:
			next_sysutte = ' *** これから{}の話をしましょう***'.format(history_theme.nowTheme)
			next_sysutte_cls = 'change_theme'
			history_utterance.add_sysutte_class(next_sysutte_cls)
			break

		if len(CANDIDATEdf) == 0:
			CANDIDATEdf = df[(df['theme'] == history_theme.nowTheme) | (df['theme'] == 'default')]
		CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
		CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]

		# 一旦選択
		zantei_df = CANDIDATEdf.sample()
		zantei_index = zantei_df.index[0]
		zantei_sysutte, zantei_theme, zantei_cls = zantei_df.values[0]

		# つかった発話は除外
		if zantei_theme != 'default':
			if zantei_sysutte in history_utterance.history_sysutte:
				CANDIDATEdf = CANDIDATEdf.drop(index=[zantei_index])
			else:
				next_sysutte, next_sysutte_cls = zantei_sysutte, zantei_cls
				history_utterance.add_sysutte(next_sysutte, zantei_theme)
				history_utterance.add_sysutte_class(next_sysutte_cls)
				correct = True
		else:
			next_sysutte, next_sysutte_cls = zantei_sysutte, zantei_cls
			history_utterance.add_sysutte(next_sysutte, 'default')
			history_utterance.add_sysutte_class(next_sysutte_cls)
			correct = True

	############ ここをメインで変更する（強化学習により）
	return next_sysutte,next_sysutte_cls



if __name__ == "__main__":

	pass















