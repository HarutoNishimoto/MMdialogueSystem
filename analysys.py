# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import re
from optparse import OptionParser
import collections
import defineClass
import sys_utterance as su
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# csvのデータ分析とかのための1回きりのコードとかたくさん

if __name__ == '__main__':
	# option
	optparser = OptionParser()
	optparser.add_option('-A', dest='action',
						help='action type', default=None, type='str')
	optparser.add_option('-C', dest='clsN',
						help='number of class', default=10, type='int')
	optparser.add_option('--cluster', dest='cluster_way',
						help='cluster way', default='KM', type='str')
	optparser.add_option('--normed', dest='normalization',
						help='normalization way', default='stand', type='str')
	(options, args) = optparser.parse_args()

	if options.action is None:
		print('No dataset filename specified, system with exit')
		sys.exit('System will exit')
	else:
		action = options.action
		clsN = options.clsN
		cluster_way = options.cluster_way
		normalization = options.normalization

	# ignore warning
	import warnings
	warnings.simplefilter("ignore")

	# params init
	params = defineClass.params()
	# userID
	df = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values
	# feature_select
	FEAdf = pd.read_csv(params.get('path_using_features'), header=None)
	fea_name = FEAdf[0].values



	if options.action == 'da':
		df = pd.read_csv(params.get('path_main_class_info'))
		X = df[['agent_utterance','cls']].values
		setX = []
		for i, val in enumerate(X):
			if val.tolist() in setX:
				pass
			else:
				setX.append(val.tolist())
		df = pd.DataFrame(data=[], columns=['agent_utterance','cls'])
		for i, val in enumerate(setX):
			df.loc[i] = val
		df = df.sort_values(['cls', 'agent_utterance'])
		df.to_csv('da5_old.csv', index=None)

	if options.action == 'dada':
		df1 = pd.read_csv('da5_old.csv')
		df2 = pd.read_csv('da5_new.csv')
		df = pd.merge(df1,df2,on='agent_utterance')
		df.to_csv('da5.csv', index=None)

	if options.action == 'scale':
		array = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
		#array = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
		Xdata = preprocessing.scale(array)#標準化（stand）
		print(np.max(Xdata) - np.min(Xdata))
		print(Xdata)


	# テーマ内の球数をカウント
	if options.action == 'cnttheme':
		df = pd.read_csv(params.get('path_theme_info'))
		print(collections.Counter(df['theme'].values))

	# コーパス内の発話の使用頻度をカウントして，0回のものは削除
	# 1回しか使用しません
	if options.action == 'cntusefreq':
		THEMEdf = pd.read_csv(params.get('path_theme_info'))
		UTTEdf = pd.read_csv('./UI3_su_da.csv')
		utte_freq = collections.Counter(UTTEdf['agent_utterance'].values)

		use_utte = THEMEdf['agent_utterance'].values
		use_freq = []
		for i, val in enumerate(use_utte):
			use_freq.append(utte_freq[val])
		THEMEdf['freq'] = use_freq
		THEMEdf = THEMEdf[['theme','freq','agent_utterance']]
		THEMEdf = THEMEdf[THEMEdf['freq'] != 0]	# 使用されなかったものは取り除く
		THEMEdf.to_csv(params.get('path_theme_info').split('.')[0] + '2.csv', index=None)

	# 疑問文とそれ以外を分離
	if options.action == 'question':
		df = pd.read_csv(params.get('path_theme_info'))
		noQuestion = df[~df['agent_utterance'].str.contains('？')]
		noQuestion.to_csv(params.get('path_theme_info').split('.')[0] + '_NOQUESTION.csv', index=None)
		Question = df[df['agent_utterance'].str.contains('？')]
		Question.to_csv(params.get('path_theme_info').split('.')[0] + '_QUESTION.csv', index=None)


	# コーパス内の発話の長さ順にソート
	if options.action == 'length':
		df = pd.read_csv(params.get('path_theme_info'))

		length = [0] * len(df)
		utte = df['agent_utterance'].values
		for i, val in enumerate(utte):
			length[i] = len(val)
		df['utte_len'] = length
		df = df.sort_values(['utte_len'])
		df.to_csv(params.get('path_theme_info').split('.')[0] + '_sort_len.csv', index=None)


	# 疑問文は依存有無でtheme_indexに差があるか
	if options.action == 'index':
		CLSdf = pd.read_csv('190926_fea8_clsInfo.csv')
		CLSdf = CLSdf[['agent_utterance','cls']]
		question_depend_o = CLSdf[CLSdf['cls'] == 4]['agent_utterance'].values 
		question_depend_x = CLSdf[CLSdf['cls'] == 3]['agent_utterance'].values

		FEAdf = pd.read_csv(params.get('path_main_class_info'))
		df = pd.merge(FEAdf, CLSdf, on='agent_utterance')

		df[df['cls_y'] == 3][['name','theme_index','agent_utterance']].to_csv('theme_index_average_x.csv', index=None)
		df[df['cls_y'] == 4][['name','theme_index','agent_utterance']].to_csv('theme_index_average_o.csv', index=None)


	# 1テーマにだけ注目したいとき
	if options.action == 'themeselect':
		theme = 'スポーツ'
		df = pd.read_csv(params.get('path_utterance_by_class'))
		SELECTdf = df[(df['theme'] == theme) | (df['theme'] == 'default')]

		print('*クラス内発話数*')
		c = collections.Counter(df['cls'].values)
		c_sorted = sorted(c.items(), key=lambda x:x[0])
		for i, val in enumerate(c_sorted):
			print(val[0],'\t',val[1])

		SELECTdf.to_csv(params.get('path_utterance_by_class').split('.')[0] + '_{}.csv'.format(theme), index=None)



	if options.action == 'tsne_view':

		df = pd.read_csv(params.get('path_main_class_info'))
		FEAdf = pd.read_csv(params.get('path_using_features'), header=None)
		use_fea = FEAdf[0].values
		X = df[use_fea].iloc[:, :].values
		X = preprocessing.scale(X)
		y = df['cls'].values
		su.TSNE_ALL(X, y)


	# tsneしたものにKM
	if options.action == 'tsne_KM':

		df = pd.read_csv(params.get('path_main_class_info'))
		FEAdf = pd.read_csv(params.get('path_using_features'), header=None)
		use_fea = FEAdf[0].values
		X = df[use_fea].iloc[:, :].values
		X = preprocessing.scale(X)

		X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
		clsInfo = su.KMeans(X_reduced, n_clusters=10)

		'''
		plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clsInfo, cmap='nipy_spectral', alpha=1)
		plt.colorbar()
		plt.show()
		'''
		df = df.rename(columns={'cls': 'cls_old'})
		df['cls'] = clsInfo
		df.to_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/191016_fea17_tsne_KM_cls10.csv')




	# PCA 表示
	if options.action == 'PCA':

		df = pd.read_csv(params.get('path_main_class_info'))
		FEAdf = pd.read_csv(params.get('path_using_features'), header=None)
		use_fea = FEAdf[0].values
		X = df[use_fea].iloc[:, :].values
		X = preprocessing.scale(X)
		clsInfo = df['cls'].values

		from sklearn.decomposition import PCA

		model = PCA(n_components=2, random_state=0)
		model.fit(X)
		X_pca = model.transform(X)
		plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clsInfo, cmap='nipy_spectral', alpha=1)
		plt.colorbar()
		plt.show()




	# 極性について
	if options.action == 'NP':

		df = pd.read_csv(params.get('path_main_class_info'))
		uttes = set(df['agent_utterance'].values)
		for u in uttes:
			su.getSentencePosNegValue(u)



	# クラスの差について
	if options.action == 'normway':

		normdf = pd.read_csv('191024_fea15_norm_bycls.csv')

		standdf = pd.read_csv('191024_fea15_stand_bycls.csv')


		df = pd.merge(normdf, standdf, on='agent_utterance')

		df.to_csv('191024_fea15_stand-norm_bycls.csv', index=None)




	if options.action == 'chk':

		df = pd.read_csv(params.get('path_main_class_info'))

		print(set(df['cls'].values))



	if options.action == 'cntutte':

		df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/191024_fea15_norm_bycls.csv')

		print(len(set(df['agent_utterance'].values)))














