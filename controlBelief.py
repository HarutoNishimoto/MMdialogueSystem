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


# 発話選択
# 特徴量のユークリッド距離を用いて選択
# POMは使用しません（2020/01/06）
def utterance_selection(user_impression, user_utterance, history_utterance, theme, candidate_action_num=1):
	# params init 
	params = defineClass.params()
	ACTdf = pd.read_csv(params.get('path_utterance_by_class'))
	THEMEdf = pd.read_csv(params.get('path_theme_info'))
	df = pd.merge(ACTdf[['agent_utterance','cls']], THEMEdf[['agent_utterance','theme']], on='agent_utterance')

	# 特定話題の選択に重み（weight）をつける
	def weightSpecificTheme(df, w=0.6):
		themes = df['theme'].values
		themes = [1-w if t == 'default' else w for t in themes]
		themes = [x/np.sum(themes) for x in themes]
		df = df.reset_index(drop=True)
		select_index = np.random.choice(df.index.values, p=themes)
		return df.loc[select_index]

	# 1交換目のとき
	if user_utterance == None:
		next_sysutte = ' *** これから{}の話をしましょう***'.format(theme)
		next_sysutte_action = 'change_theme'
		history_utterance.add_sysutte(next_sysutte, next_sysutte_action)
		return next_sysutte, next_sysutte_action
	else:
		## ユークリッド距離計算
		# クラスごとの代表値を読み込み 
		feadf = pd.read_csv(params.get('path_feature_average_by_class'))
		feadf['prevUI3'] = feadf['UI3average'].values - feadf['UI3average_diff'].values
		fea_ave = feadf[['prevUI3', 'num_noun_before', 'u_utte_len_before']].values
		# 現在の特徴量を計算
		num_noun_cnt, u_utte_len_cnt = getNounandLen(user_utterance)
		fea_cnt = [[user_impression, num_noun_cnt, u_utte_len_cnt]]
		ss = preprocessing.StandardScaler()
		ss.fit(fea_ave)
		ave_ss = ss.transform(fea_ave)
		cnt_ss = ss.transform(fea_cnt)
		# 距離を計算
		action = np.sort(list(set(df['cls'].values)))
		distance = []
		for i, val in enumerate(action):
			dist = np.linalg.norm(cnt_ss - ave_ss[i])
			distance.append(dist)
		# 一番近いクラスを選択
		Dict = dict(zip(action, distance))
		candidate_action = []
		for k, v in sorted(Dict.items(), key=lambda x: x[1])[:candidate_action_num]:
			candidate_action.append(k)
		n_next_action = np.random.choice(candidate_action)

		# 発話の選択
		CANDIDATEdf = df[(df['cls'] == n_next_action) &
			((df['theme'] == theme) | (df['theme'] == 'default'))]
		CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
		CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
		# 使えないものを削除
		for i in range(len(CANDIDATEdf)):
			if CANDIDATEdf.loc[i, :]['agent_utterance'] in history_utterance.history_sysutte:
					CANDIDATEdf = CANDIDATEdf.drop(index=[i])
		# 候補が残っていないなら，action気にせず候補を決定
		if len(CANDIDATEdf) == 0:
			CANDIDATEdf = df[(df['theme'] == theme) | (df['theme'] == 'default')]
			CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
			CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
			# 使えないものを削除
			for i in range(len(CANDIDATEdf)):
				if CANDIDATEdf.loc[i, :]['agent_utterance'] in history_utterance.history_sysutte:
						CANDIDATEdf = CANDIDATEdf.drop(index=[i])
		# 選択して終了
		SELECTdf = weightSpecificTheme(CANDIDATEdf)
		next_sysutte, next_theme, next_action = SELECTdf.values
		history_utterance.add_sysutte(next_sysutte, next_action)

		return next_sysutte, next_action




if __name__ == "__main__":

	pass















