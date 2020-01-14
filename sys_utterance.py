# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import re
from optparse import OptionParser
import collections
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
import defineClass
import getSysUtteranceFea as SUF


# KMeans
def KMeans(Xdata, n_clusters=10):
	from sklearn.cluster import KMeans
	clsInfo = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(Xdata)
	return clsInfo

def TSNE_ALL(X, y):
	from sklearn.manifold import TSNE
	X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
	plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='nipy_spectral', alpha=1)
	plt.colorbar()
	plt.show()

# dfから不適切な行を削除（列'label'の'E'や' 'を排除）
def remDataFrameError(df, meta='label', remove=True, devision=False):
	df[meta] = df[meta].astype('str')
	df[meta] = df[meta].replace(r'\D', '', regex=True)
	if remove == True:
		df = df[df[meta] != '']
		df[meta] = df[meta].astype('int')
	if devision == True:
		df[meta] = df[meta].values /10.0
	return df

# 特徴量の平均をクラスごとに計算してファイルに出力
def getFeatureAverageBycls(params, fea_name):
	print('#####分けられたクラスごとの特徴量を計算して書き出し#####')
	CLSdf = pd.read_csv(params.get('path_main_class_info'))
	CLSdf = CLSdf[fea_name.tolist() + ['cls']]
	CLSdf = CLSdf.sort_values(['cls'])

	clsname = set(CLSdf['cls'].values)
	feaaveInfo = []
	for c in clsname:
		df_cnt = CLSdf[CLSdf['cls'] == c]
		fea = df_cnt.iloc[:, :-1].values
		feaave = np.mean(fea, axis=0).tolist()
		feaaveInfo.append(feaave)

	feaavedf = pd.DataFrame(data=feaaveInfo, columns=fea_name)
	feaavedf.to_csv(params.get('path_feature_average_by_class'), index=None)


# クラスに含まれる発話をファイルに出力
def getUtteranceBycls(params):
	print('#####分けられたクラスごとにシステム発話を書き下し#####')
	CLSdf = pd.read_csv(params.get('path_main_class_info'))
	meta = ['cls', 'freq', 'freq_1cls', 'ratio', 'agent_utterance']
	SUdf = pd.DataFrame(data=[], columns=meta)
	clas = list(set(CLSdf['cls'].values))
	num = 0

	freqDict = collections.Counter(CLSdf['agent_utterance'].values)# 頻度の辞書
	for c in clas:
		df_1cls = CLSdf[CLSdf['cls'] == c]
		agent_utterance_1cls = list(set(df_1cls['agent_utterance'].values))
		freqDict_1cls = collections.Counter(df_1cls['agent_utterance'].values)

		for val in agent_utterance_1cls:
			SUdf.loc[num] = [c, freqDict[val], freqDict_1cls[val], round(float(freqDict_1cls[val])/freqDict[val], 2), val]
			num += 1

	THEMEdf = pd.read_csv(params.get('path_theme_info'))
	THEMEdf = THEMEdf.drop('freq', axis=1)
	THEMEdf = THEMEdf.drop('cls', axis=1)#####ここは書き換えありかも
	SUdf = pd.merge(SUdf, THEMEdf, on='agent_utterance')

	df_sorted = SUdf.sort_values(['cls', 'theme'])
	df_sorted = df_sorted.drop_duplicates()
	df_sorted.to_csv(params.get('path_utterance_by_class'), index=None)



# クラスごとの発話の数をカウントしてファイルに出力
def getFreqBycls(params):
	print('#####クラスタ内テーマごとの発話カウント#####')
	THEMEdf = pd.read_csv(params.get('path_theme_info'))
	theme = np.sort(list(set(THEMEdf['theme'].values)))
	SUdf = pd.read_csv(params.get('path_utterance_by_class'))
	clas = sorted(list(set(SUdf['cls'].values)))

	DAdict = {}
	for i, c in enumerate(clas):
		DAdict[c] = i

	info = np.zeros((len(clas), len(theme)))
	for su, c in zip(SUdf['agent_utterance'].values, SUdf['cls'].values):
		theme_cnt = THEMEdf[THEMEdf['agent_utterance'] == su]['theme'].values
		for t in theme_cnt:
			info[DAdict[c], np.where(theme == t)[0]] += 1

	freq_1class = []
	for val in info:
		freq_1class.append(np.sum(val))
	CLS_THEME_df = pd.DataFrame(data=info, columns=theme)
	CLS_THEME_df['all'] = freq_1class
	CLS_THEME_df.to_csv(params.get('path_freq_class_theme'), index=None)



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

	# param init
	params = defineClass.params()
	clsN = int(params.get('class_num'))
	# userID
	df = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values
	# feature_select
	FEAdf = pd.read_csv(params.get('path_using_features'))
	fea_name = FEAdf['feature'].values
	fea_norm = FEAdf['norm'].values
	fea_weight = FEAdf['weight'].values


	# 特徴量を変更するたびに実行
	# ベースファイル作成
	if action == 'extfea_clustering':
		Uttedf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/UI3_su_da.csv')
		dialogFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/dialogue.csv')
		textFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/text.csv')
		Uttedf = Uttedf[['name','agent_utterance']]
		df = pd.merge(dialogFeadf[['name', 'lenS', 'label']], textFeadf[['name', '名詞']], on='name')
		df = pd.merge(df, Uttedf, on='name')
		df = remDataFrameError(df, meta='label', remove=True, devision=True)

		# そのためにテーマ情報のファイルとmergeすることで，使用しないシステム発話をあらかじめ取り除く
		# 取り除いてからKMなりのクラスタリングをする
		THEMEdf = pd.read_csv(params.get('path_theme_info'))
		df = pd.merge(df, THEMEdf, on='agent_utterance')
		# 表記揺れを解消
		df['agent_utterance_tmp'] = df['agent_utterance'].values
		df['agent_utterance'] = df['agent_utterance_refine'].values

		### feature作成 ####
		df = SUF.getSpecificWord(df)					#dim3
		df = SUF.getIfQuestion(df)						#dim1
		df = SUF.getThemeOnehot(df)						#dim5
		#df = SUF.getGeneralityInfo(df, userID_1902)		#dim1
		#df = SUF.getContextInfo(df)						#dim1

		dfs = []
		for ID in userID_1902:
			print(ID)
			df_cnt = df[df['name'].str.startswith(ID)]
			#### feature作成 ####
			df_cnt = SUF.getUserImpression(df_cnt)		#dim2
			#df_cnt = SUF.getUserUtteLength(df_cnt, ID)	#dim2
			#df_cnt = SUF.getNumNoun(df_cnt)				#dim2
			#df_cnt = SUF.getUtteranceThemeIndex(df_cnt)	#dim1

			dfs.append(df_cnt)
		df = pd.concat(dfs)



		# 特徴量ごとに正規化（ファイルに書かれている通りに）
		FEAdf = pd.DataFrame(data=[], columns=[])
		for i, (f, n, w) in enumerate(zip(fea_name, fea_norm, fea_weight)):
			if n == 'stand':
				FEAdf[f] = float(w) * preprocessing.scale(df[f].values)
			elif n == 'norm':
				FEAdf[f] = float(w) * preprocessing.minmax_scale(df[f].values)
			else:
				FEAdf[f] = float(w) * df[f].values

		# クラスタリング
		clsInfo = KMeans(FEAdf.iloc[:,:].values, n_clusters=clsN)
		df['cls'] = clsInfo
		# 表記揺れを解消
		df['agent_utterance'] = df['agent_utterance_tmp'].values
		df = df.drop('agent_utterance_tmp', axis=1)
		df.to_csv(params.get('path_main_class_info'), index=None)


	# 必要なファイルをまとめて作成
	if action == 'makefile':
		getFeatureAverageBycls(params, fea_name)
		getUtteranceBycls(params)
		getFreqBycls(params)


	# 手で作った対話行為を元にベースファイル生成
	if action == 'clustering_handda':
		Uttedf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/200109_cls8.csv')#name,agent_utterance
		dialogFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/dialogue.csv')
		textFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/text.csv')
		FEAdf = pd.merge(dialogFeadf[['name', 'lenS', 'label']], textFeadf[['name', '名詞']], on='name')
		df = pd.merge(FEAdf, Uttedf[['name','agent_utterance']], on='name')
		df = remDataFrameError(df, meta='label')

		dfs = []
		for ID in userID_1902:
			print(ID)
			df_cnt = df[df['name'].str.startswith(ID)]
			#### feature作成 ####
			df_cnt = SUF.getUserImpression(df_cnt)		#dim2
			df_cnt = SUF.getUserUtteLength(df_cnt, ID)		#dim2
			df_cnt = SUF.getNumNoun(df_cnt)				#dim2

			dfs.append(df_cnt)
		df = pd.concat(dfs)

		CLSdf = pd.read_csv('./baseDA10.csv')#agent_utterance,cls
		df = pd.merge(df, CLSdf[['agent_utterance','cls']], on='agent_utterance')
		df.to_csv(params.get('path_main_class_info'), index=None)





