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


if __name__ == '__main__':
	# option
	optparser = OptionParser()
	optparser.add_option('-A', dest='action',
						help='action type', default=None, type='str')
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
	#df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/refData/userID_1902.txt', header=None)
	#userID_1902 = df[0].values
	# param get
	params = defineClass.params()


	# クラスを心象により細分化
	if options.action == 'clsfyUI':

		FEANAMEdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/refData/191024_fea15.csv', header=None)
		feaname = FEANAMEdf[0].values.tolist()

		DAdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/1902themeInfo/1902MMcorpus_theme_191212.csv')
		FEAdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/191024_fea15_norm.csv')

		UIdf = pd.DataFrame(data=[], columns=['agent_utterance', 'theme']+feaname)
		utte = DAdf[DAdf['da'] == 'qs_nocont'][['agent_utterance', 'theme']].values
		subdivide = {}
		for i, (u, t) in enumerate(utte):
			feaave = np.average(FEAdf[FEAdf['agent_utterance'] == u][feaname].values, axis=0)
			UIdf.loc[i] = [u, t] + feaave.tolist()
			if feaave[8] >= 5:# ここがメイン（心象[8]が閾値（5）より高芋の、それ以外のものとする）
				subdivide[u] = 'high'
			else:
				subdivide[u] = 'normal'

		da_subdivide = DAdf['da'].values
		for i, u in enumerate(DAdf['agent_utterance'].values):
			if u in subdivide.keys():
				da_subdivide[i] = da_subdivide[i] + '_' + subdivide[u]

		DAdf['da'] = da_subdivide
		DAdf.to_csv('aaa.csv', index=None)




	# システム発話ごとに簡単な対話行為を定義して，ファイル作成
	# これはおそらく終わりまで固定なきがする
	if options.action == 'simpleDA':
		df = pd.read_csv(params.get('path_theme_info'))
		su = df['agent_utterance'].values
		su_ref = df['agent_utterance_refine'].values
		clas = df['cls'].values

		DAdf = pd.DataFrame(data=[], columns=['agent_utterance','agent_utterance_refine','da_simple'])
		for i, (s1, s2, c) in enumerate(zip(su, su_ref, clas)):
			if c == 'io':
				DAdf.loc[i] = [s1, s2, 'io']
			elif '？' in s2:
				DAdf.loc[i] = [s1, s2, 'qs']
			else:
				DAdf.loc[i] = [s1, s2, 're']

		DAdf.to_csv('simpleDA.csv', index=None)


	# ベースライン手法で使用する，対話行為を設計
	# 複雑すぎるのでもう少し改良
	if options.action == 'baseDA':
		import collections

		df = pd.read_csv(params.get('path_theme_info'))
		baseDA = [''] * len(df)

		for i in range(len(df)):
			if '？' in df.loc[i,'agent_utterance_refine']:
				baseDA[i] = 'qs'
			elif 'io' == df.loc[i,'cls']:
				baseDA[i] = 'io'
			elif 'ありがとう' in df.loc[i,'agent_utterance']:
				baseDA[i] = 'thank'
			else:
				baseDA[i] = 're'

		df['cls'] = baseDA
		df_sorted = df.sort_values(['cls', 'theme', 'agent_utterance_refine'])
		df_sorted = df_sorted[['cls','theme','agent_utterance','agent_utterance_refine']]
		df_sorted.to_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/200110_baseDA6.csv', index=None)




	# 作成した対話行為の評価
	# 1対話行為に役割の異なる対話行為がどれくらい含まれているかを表示する
	if options.action == 'evaDA':
		AUTOdf = pd.read_csv(params.get('path_utterance_by_class'))
		HANDdf = pd.read_csv(params.get('path_theme_info'))#cls_y
		handDA = HANDdf['cls'].values
		for i, da in enumerate(handDA):# 書き換え
			if 'qs_nocont' in da:
				handDA[i] = 'qs_nocont'
			if da in ['re_music','re_trip','re_sport','re_food']:
				handDA[i] = 're_specific'
		HANDdf['cls'] = handDA

		df = pd.merge(AUTOdf, HANDdf, on='agent_utterance')
		autoDA = list(set(AUTOdf['cls'].values))
		for da in autoDA:
			print(da)
			print(collections.Counter(df[df['cls_x'] == da]['cls_y'].values))



	if options.action == 'calaveUI':
		UTTEdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/200110_baseDA4.csv')
		UIdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/200109_cls8.csv')
		df = pd.merge(UTTEdf, UIdf, on='agent_utterance')

		da_simple = set(df['cls_x'].values)

		for da in da_simple:
			UIave_cnt = df[df['cls_x'] == da]['UI3average'].values
			print(da, len(UIave_cnt), np.mean(UIave_cnt))










