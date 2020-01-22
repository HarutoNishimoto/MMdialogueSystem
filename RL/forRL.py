# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
from optparse import OptionParser
import collections
from sklearn import preprocessing
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import matplotlib.pyplot as plt
import pickle
from el_agent import softmax



if __name__ == '__main__':
	# option
	optparser = OptionParser()
	optparser.add_option('-A', dest='action',
						help='action type', default=None, type='str')
	optparser.add_option('-I', dest='input',
						help='input', default=None, type='str')
	optparser.add_option('-T', dest='type',
						help='type', default=None, type='str')
	(options, args) = optparser.parse_args()

	if options.action is None:
		print('No dataset filename specified, system with exit')
		sys.exit('System will exit')
	else:
		action = options.action
	# ignore warning
	import warnings
	warnings.simplefilter("ignore")
	# userID
	df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values
	# param get
	params = defineClass.params()


	# name, sys_utterance, user_utterance, UI3, UI3_diffを並べたファイルを作成
	# ユーザモデル作成のために必要（1回きり）
	if options.action == 'makeExchgUIFile':
		# s_utte,u_utte
		EXCHGdf = pd.DataFrame(data=[], columns=['name','sys_utterance','user_utterance'])
		index = 0
		for ID in userID_1902:
			Sysdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_S.txt'.format(ID), header=None)
			Userdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_U.txt'.format(ID), header=None)
			sys_utterance = Sysdf[0].values
			user_utterance = Userdf[0].values
			for i, (s, u) in enumerate(zip(sys_utterance, user_utterance)):
				EXCHGdf.loc[index] = ['{}_{}'.format(ID, str(i+1).zfill(3)), s, u]
				index += 1
		# UI3
		UIdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/191024_fea15_norm.csv')
		UIdf = UIdf[['name','UI3average','UI3average_diff']]
		# 結合
		df = pd.merge(EXCHGdf, UIdf, on='name')
		df = df[~df.duplicated()]
		df.to_csv('exchgUI3Info.csv', index=None)


	# mecabが取れる名詞を確認
	if action == 'getnoun':
		mt = MeCab.Tagger()

		for ID in userID_1902:
			df1 = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_U.txt'.format(ID), header=None)
			df2 = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_S.txt'.format(ID), header=None)
			utte1 = df1[0].values
			utte2 = df1[0].values
			utte = utte1 + utte2

			for i, val in enumerate(utte):
				node = mt.parseToNode(val)
				while node:
					fields = node.feature.split(",")
					#print(fields)
					# 名詞、動詞、形容詞に限定
					if fields[0] == '名詞':
						print(node.surface, fields[1], fields[2], fields[3])
					node = node.next




	# クラスに名前つけ
	if options.action == 'naming':
		df = pd.read_csv(params.get('path_utterance_by_class'))
		NAMEdf = pd.read_csv(params.get('path_class_name'))
		newCLS = [''] * len(df)
		for i, c in enumerate(df['cls'].values):
			newCLS[i] = NAMEdf[NAMEdf['clsKM'] == c]['clsname'].values[0]

		df = df.rename(columns={'cls': 'clsKM'})
		df['cls'] = newCLS
		df.to_csv(params.get('path_utterance_by_class').split('.')[0] + '_named.csv', index=None)


	# parameters.txtを書き換え
	if options.action == 'rewrite':
		keyword = ['path_utterance_by_class_named','path_class_name','class_num']
		with open('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/refData/parameters.txt', 'r')as rf:
			lines = rf.readlines()

		for i, val in enumerate(lines):
			for key in keyword:
				if key in lines[i]:
					if lines[i].startswith('###'):
						lines[i] = lines[i].replace('###','')
					else:
						lines[i] = '###' + lines[i]

					print(lines[i])

		with open('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/refData/parameters.txt', 'w')as wf:
			for l in lines:
				wf.write(l)



	# 学習回数をテーブルの要素ごとに表示
	if options.action == 'Qval':
		# 学習すみQテーブルの読み込み
		with open('{}/{}_{}'.format(options.input, options.input, options.type), mode='rb') as f:
			Q = pickle.load(f)

		# 辞書を2次元リストに変換
		state_size, action_size = 24, 4
		reward_map = np.zeros((state_size, action_size))
		for s in range(state_size):#state_size
			for a in range(action_size):#action_size
				if s in Q.keys():
					reward_map[s][a] = Q[s][a]

		df = pd.DataFrame(data=reward_map, columns=range(action_size), index=range(state_size))
		df.to_csv('{}/{}_{}.csv'.format(options.input, options.input, options.type))













	# coef_epsilonについての検討
	if options.action == 'ep':
		randomtime = 0
		for e in range(500):
			epsilon = (0.90 ** e)
			randomtime += epsilon * 10
			print(e, epsilon, randomtime)

			if epsilon < 0.1:
				print(e, epsilon)
				break


	# softmaxすることによる効果を検証
	if options.action == 'softmax':
		# 学習すみQテーブルの読み込み
		with open('{}/{}_Q'.format(options.input, options.input), mode='rb') as f:
			Q = pickle.load(f)

		# 辞書を2次元リストに変換
		state_size, action_size = 24, 8
		reward_map = np.zeros((state_size, action_size))
		for s in range(state_size):#state_size
			for a in range(action_size):#action_size
				if s in Q.keys():
					reward_map[s][a] = Q[s][a]

		###
		reward_map_normed = preprocessing.minmax_scale(reward_map, axis=1)
		df = pd.DataFrame(data=reward_map_normed, columns=range(action_size), index=range(state_size))
		df.to_csv('{}/{}_normed.csv'.format(options.input, options.input))

		###
		aaa = reward_map
		for i, val in enumerate(aaa):
			aaa[i] = softmax(val, coef=1)
		df = pd.DataFrame(data=aaa, columns=range(action_size), index=range(state_size))
		df.to_csv('{}/{}_smax.csv'.format(options.input, options.input))

		###
		aaa = reward_map_normed
		for i, val in enumerate(aaa):
			aaa[i] = softmax(val, coef=1)
		df = pd.DataFrame(data=aaa, columns=range(action_size), index=range(state_size))
		df.to_csv('{}/{}_norm_smax.csv'.format(options.input, options.input))



	# softmaxの係数についての検討
	if options.action == 'exsoft':
		# 学習すみQテーブルの読み込み
		with open('{}/{}'.format(options.input, options.input), mode='rb') as f:
			Q = pickle.load(f)

		# 辞書を2次元リストに変換
		state_size, action_size = 24, 8
		reward_map = np.zeros((state_size, action_size))
		for s in range(state_size):#state_size
			for a in range(action_size):#action_size
				if s in Q.keys():
					reward_map[s][a] = Q[s][a]

			print(softmax(preprocessing.minmax_scale(reward_map[s]), coef=int(options.type))*100)









