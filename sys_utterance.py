# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import re
from optparse import OptionParser
import collections
import MeCab
import jaconv
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
import gensim
import defineClass



# 文章のポジネガを判定するマン
def getSentencePosNegValue(sentence):
	df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/pn_ja.txt',
		sep=':',
		names=('org','kana','pos','value'))
	use_poses = ['形容詞']
	mt = MeCab.Tagger()
	all_pn_val, cnt = 0, 0
	node = mt.parseToNode(sentence)
	while node:
		fields = node.feature.split(",")
		if (fields[0] in use_poses) and (len(fields) == 9):
			if fields[6] in df['org'].values:
				word_pn_val = df[df['org'] == fields[6]]['value'].values
				word_pn_val = np.mean(word_pn_val)
			else:
				kana = jaconv.kata2hira(fields[7])
				word_pn_val = df[df['kana'] == kana]['value'].values
				word_pn_val = np.mean(word_pn_val)
			if not np.isnan(word_pn_val):
				all_pn_val += word_pn_val
				cnt += 1
		node = node.next

	if cnt != 0:
		print(sentence+','+str(all_pn_val/cnt))
		return all_pn_val/cnt
	else:
		print(sentence+','+str(0))
		return 0



# dfに特徴量増やしてreturn
# 190926の対話行為的な特徴量を追加する
def getUtteranceThemeIndex(df):
	startUtterance = pd.read_csv('./1902themeInfo/1902MMcorpus_theme_start_utterance.csv')['agent_utterance'].values
	theme_index = []
	countup = 0
	utterances = df['agent_utterance'].values
	for utte in utterances:
		if utte in startUtterance:
			theme_index.append(0)
			countup = 1
		else:
			theme_index.append(countup)
			countup += 1
	df['theme_index'] = theme_index

	return df

# dfに特徴量増やしてreturn
def getDialogueAct(df):
	DAdf = pd.read_csv('./dialogue_act_5.csv')
	DAinfo = DAdf['cls'].values
	DAdict = {}
	da = list(sorted(set(DAinfo)))
	for i, val in enumerate(da):
		DAdict[val] = i
	DAinfo = [DAdict[x] for x in DAinfo]
	DA_onehot = np.eye(len(da))[DAinfo]
	for i, m in enumerate(da):
		DAdf[m] = DA_onehot[:, i]
	NEWdf = pd.merge(df, DAdf[['name'] + da], on='name', how='left')
	NEWdf = NEWdf.fillna(0)

	return NEWdf

# dfに特徴量増やしてreturn
# 特定の単語（文脈依存の単語）が含まれているか
def getContextDependentWord(df):
	sysUtte = df['agent_utterance'].values

	WORDdf = pd.read_csv('./../refData/contextDependentWords.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['context_dependent_word'] = CDfea
	return df

# dfに特徴量増やしてreturn
# 特定の単語（文脈依存の単語）が含まれているか
# 指示語と相槌を別の特徴量にしました
def getDependNodWord(df):
	sysUtte = df['agent_utterance'].values

	WORDdf = pd.read_csv('./../refData/dependWord.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['depend_word'] = CDfea

	WORDdf = pd.read_csv('./../refData/nodWord.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['nod_word'] = CDfea

	return df



# dfに特徴量増やしてreturn
# システム発話は文脈依存かどうか
def getContextInfo(df):
	sysUtte = df['agent_utterance'].values
	contextDict = {}
	for su in list(set(sysUtte)):
		contextDict[su] = {}

	for i, val in enumerate(sysUtte[:-1]):
		if val in contextDict[sysUtte[i+1]]:
			contextDict[sysUtte[i+1]][val] += 1
		else:
			contextDict[sysUtte[i+1]][val] = 1

	context = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		context[i] = len(contextDict[val])
	df['context'] = context
	return df


# dfに特徴量増やしてreturn
# 汎用性のあるシステム発話かどうか
def getGeneralityInfo(df, userID):
	sysUtte = df['agent_utterance'].values

	reuseDict = {}
	for ID in userID:
		df_cnt = df[df['name'].str.startswith(ID)]
		sysutte = df_cnt['agent_utterance'].values
		freq = collections.Counter(sysutte)

		for k, v in freq.items():
			if k in reuseDict:
				reuseDict[k].append(v)
			else:
				reuseDict[k] = [v]

	general = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		general[i] = np.mean(reuseDict[val])
	df['general'] = general
	return df


# dfに特徴量増やしてreturn
# ユーザの発話長，直前と直後のもの
# 音声認識の不具合がおおいので，語数ではなく発話時間として近似している（2019/09/11）
def getUserUtteLength(df, userID):
	TSdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/timestamp/{}.csv'.format(userID))
	index = TSdf['ts'].values
	index = [userID + '_' + x.replace('ts','').zfill(3) for x in index]
	TSdf['name'] = index
	TMPdf = pd.merge(df, TSdf, on='name')

	s_end = TMPdf['s_end'].values
	u_end = TMPdf['u_end'].values

	u_utte_len_after = u_end - s_end
	u_utte_len_before = np.array([0] + u_utte_len_after[:-1].tolist())
	df['u_utte_len_after'] = u_utte_len_after / 1000 * 5
	df['u_utte_len_before'] = u_utte_len_before / 1000 * 5
	return df

# dfに特徴量増やしてreturn
# システム発話長の特徴量
def getSysUtteLength(df):
	lenS = df['lenS'].values
	lenS_diff = [0] + np.diff(lenS).tolist()
	df['lenS_diff'] = lenS_diff
	return df

# dfに特徴量増やしてreturn
# 前後の発話の名詞の数
def getNumNoun(df):
	num_noun_after = df['名詞'].values
	num_noun_before = [0] + num_noun_after[:-1].tolist()
	df['num_noun_after'] = num_noun_after
	df['num_noun_before'] = num_noun_before
	return df

# dfに特徴量増やしてreturn
# UIについての特徴量
def getUserImpression(df):
	UI3average = df['label'].values
	UI3average_diff = [0] + np.diff(UI3average).tolist()
	df['UI3average'] = UI3average
	df['UI3average_diff'] = UI3average_diff
	return df

# 表層的な特徴量（？が入っているかどうか）
def getIfQuestion(df):
	sysUtte = df['agent_utterance'].values
	# 入っている:1，入っていない:0
	sysUtte = [1 if '？' in x else 0 for x in sysUtte]
	df['questionmark'] = sysUtte
	return df

# テーマの特徴量（テーマ数の長さ(16)）
def getUtteTheme(df):
	THEMEdf = pd.read_csv('./../refData/1902MMcorpus_theme.txt')
	themes = np.sort(list(set(THEMEdf['theme'].values)))
	sysUtte = df['agent_utterance'].values
	mat = np.zeros((len(sysUtte), len(themes)))
	for i, val in enumerate(sysUtte):
		theme = THEMEdf[THEMEdf['agent_utterance'] == val]['theme'].values
		if len(theme) == 0:
			mat[i] = np.zeros(len(themes))
		else:
			mat[i] = np.eye(len(themes))[np.where(themes == theme)]

	for i, val in enumerate(themes):
		df[val] = mat[:, i]
	return df

# テーマ情報のparse
def makeUtteTheme(filename):
	with open(filename, 'r', encoding="shift_jis")as f:
		utterances = f.readlines()

	utterances = [x.replace('\n', '') for x in utterances]
	utterances = [x for x in utterances if x != '']

	cnt_theme = ''
	for i, val in enumerate(utterances):
		if val.startswith('%'):
			cnt_theme = val
		else:
			utterances[i] = val + ',' + cnt_theme

	with open('ThemeInfo.txt', 'w', encoding="utf-8")as f:
		for ele in utterances:
			f.write(ele + '\n')

# 入力は1文
# https://qiita.com/kenta1984/items/93b64768494f971edf86
# テキストのベクトルを計算
def getSentenceVector(sentence, model, vec_size=300):
	mt = MeCab.Tagger()
	vocab = model.wv.vocab

	sum_vec = np.zeros(vec_size)
	word_count = 0
	node = mt.parseToNode(sentence)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				sum_vec += model.wv[node.surface]
				word_count += 1
		node = node.next

	return sum_vec / word_count

def EmbeddingAverage(sentence, model, vec_size=300):
	mt = MeCab.Tagger()
	vocab = model.wv.vocab

	sum_vec = np.zeros(vec_size)
	node = mt.parseToNode(sentence)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				sum_vec += model.wv[node.surface]
		node = node.next

	length = np.linalg.norm(sum_vec,2)
	return sum_vec / length


# targetの文のembed_vec出す．
def VectorExtrema(sent_target, sent_compare, model, vec_size=300):
	mt = MeCab.Tagger()
	vocab = model.wv.vocab

	wv_target = []
	node = mt.parseToNode(sent_target)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				wv_target.append(model.wv[node.surface].tolist())
		node = node.next
	wv_target = np.array(wv_target)

	wv_compare = []
	node = mt.parseToNode(sent_compare)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				wv_compare.append(model.wv[node.surface].tolist())
		node = node.next
	wv_compare = np.array(wv_compare)

	wv_target_max = np.max(wv_target, axis=0)
	wv_target_min = np.min(wv_target, axis=0)
	wv_compare_abs_min = np.abs(np.min(word_vec, axis=0))

	ret_vec = np.zeros((1, vec_size))
	for i in range(vec_size):
		if wv_target_max[i] > wv_compare_abs_min[i]:
			ret_vec[i] = wv_target_max[i]
		else:
			ret_vec[i] = wv_target_min[i]
	return ret_vec

# targetの文のembed_vec出す．
# 作成の途中段階
def GreedyMatching(sent_target, sent_compare, model, vec_size=300):
	mt = MeCab.Tagger()
	vocab = model.wv.vocab

	wv_target = []
	node = mt.parseToNode(sent_target)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				wv_target.append(model.wv[node.surface].tolist())
		node = node.next
	wv_target = np.array(wv_target)

	wv_compare = []
	node = mt.parseToNode(sent_compare)
	while node:
		fields = node.feature.split(",")
		# 名詞、動詞、形容詞に限定
		if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
			if node.surface in vocab:
				wv_compare.append(model.wv[node.surface].tolist())
		node = node.next
	wv_compare = np.array(wv_compare)

	sum_vec = []
	for i, tar in enumerate(wv_target):
		for j, com in enumerate(wv_compare):
			find_max(cos_sim(tar, com))

		sum_vec += max(cos_sim(tar, com))

	length = np.linalg.norm(sent_target, 2)
	Grr = sum_vec/sent_target

	# 逆もする

	wv_target_max = np.max(wv_target, axis=0)
	wv_target_min = np.min(wv_target, axis=0)
	wv_compare_abs_min = np.abs(np.min(word_vec, axis=0))

	ret_vec = np.zeros((1, vec_size))
	for i in range(vec_size):
		if wv_target_max[i] > wv_compare_abs_min[i]:
			ret_vec[i] = wv_target_max[i]
		else:
			ret_vec[i] = wv_target_min[i]
	return ret_vec

# https://qiita.com/kenta1984/items/93b64768494f971edf86
# cos類似度を計算
def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 対象発話の前N発話までの類似度をそれぞれ計算
def getSimilarity(df, model, N=3, vec_size=300):
	sysUtte = df['agent_utterance'].values

	noun = [0] * len(sysUtte)
	for i, utte in enumerate(sysUtte):
		#noun[i] = getSentenceVector(utte, model)
		noun[i] = EmbeddingAverage(utte, model)

	noun = [np.zeros(vec_size)] * (N-1) + noun
	similarity = [0] * len(sysUtte)
	for i in range(len(sysUtte)):
		sims = []
		for j in range(N):
			sims.append(cos_sim(noun[i+N-1], noun[i+N-1-(j+1)]))
		similarity[i] = sims
	similarity = np.array(similarity)
	# append
	for i in range(N):
		sim_cnt = similarity[:, i]
		sim_cnt = [0 if np.isnan(x) else x for x in sim_cnt]
		df['sim{}'.format(str(i+1))] = sim_cnt
	return df

# ルールベースでクラスタリング
def RuleBaseClustering(Xdata, meta):
	df = pd.DataFrame(data=Xdata, columns=meta)
	question = (df['questionmark'] == 1).values
	depend = (df['depend_word'] == 1).values
	nod = (df['nod_word'] == 1).values
	s_len = (df['lenS'] >= 30).values
	fea_bin = np.array([question, depend, nod, s_len]).transpose()
	fea_bin = np.where(fea_bin == True, 'o', 'x')

	RULEdf = pd.read_csv('./../refData/rule_NH.csv')
	clsInfo = [''] * len(df)
	for i, val in enumerate(fea_bin):
		for c, rule in zip(RULEdf['cls'].values, RULEdf.iloc[:, :-1].values):
			if list(val) == list(rule):
				clsInfo[i] = c
	return clsInfo

# KMeans
def KMeans(Xdata, n_clusters=10):
	from sklearn.cluster import KMeans
	clsInfo = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(Xdata)
	return clsInfo

# KMを2段階でするもの
def KMeans_2step(Xdata, n_clusters1=5, n_clusters2=2):
	clsInfo_1step = KMeans(Xdata, n_clusters=n_clusters1)
	classed_Xdata_1step = np.hstack((Xdata, np.array([clsInfo_1step]).transpose()))
	df_1step = pd.DataFrame(data=classed_Xdata_1step, columns=fea_name.tolist()+['cls'])
	dfs = []
	for c in set(clsInfo_1step):
		df_cnt = df_1step[df_1step['cls'] == c]
		Xdata_2step_cnt = df_cnt.iloc[:, 1:-1].values
		clsInfo_2step_cnt = KMeans(Xdata_2step_cnt, n_clusters=n_clusters2)
		clsInfo_2step_cnt = [str(c) + '_' + str(x) for x in clsInfo_2step_cnt]
		df_cnt['cls'] = clsInfo_2step_cnt
		dfs.append(df_cnt)
	df_2step = pd.concat(dfs)
	return df_2step['cls'].values


# 階層的クラスタリング
def HierarchicalClustering(Xdata, n_clusters=3, method='ward', metric='euclidean'):
	from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
	Z = linkage(Xdata, 
				metric = metric, 
				method = method)

	clsInfo = fcluster(Z, n_clusters, criterion='maxclust') # クラスタ数で分けたい場合
	return clsInfo


def TSNE_ALL(X, y):
	from sklearn.manifold import TSNE
	X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
	plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='nipy_spectral', alpha=1)
	#plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='gist_rainbow', alpha=1)
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

def RMSE(true, pred, printing=True):
	RMSE = np.sqrt(np.mean((true-pred)**2))
	if printing == True:
		print('RMSE', RMSE)
	return RMSE

#### 分析をするためのスクリプト ####

if __name__ == '__main__':
	# option
	optparser = OptionParser()
	optparser.add_option('-A', dest='action',
						help='action type', default=None, type='str')
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
		cluster_way = options.cluster_way
		normalization = options.normalization

	# ignore warning
	import warnings
	warnings.simplefilter("ignore")

	# param init
	params = defineClass.params()
	clsN = int(params.get('class_num'))
	# clf
	clf = svm.SVR(kernel='rbf', gamma='auto')
	# userID
	df = pd.read_csv('./../refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values
	# feature_select
	FEAdf = pd.read_csv(params.get('path_using_features'), header=None)
	fea_name = FEAdf[0].values





	# 特徴量を変更するたびに実行
	if action == 'extfea_clustering':
		Uttedf = pd.read_csv('./UI3_su_da.csv')
		dialogFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/dialogue.csv')
		textFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/text.csv')
		Uttedf = Uttedf[['name','agent_utterance','dialogue_act']]
		df = pd.merge(dialogFeadf[['name', 'lenS', 'label']], textFeadf[['name', '名詞']], on='name')
		df = pd.merge(df, Uttedf, on='name')
		df = remDataFrameError(df, meta='label', remove=True, devision=True)

		# https://github.com/Kyubyong/wordvectors ここの日本語w2vモデルを使用
		#model = gensim.models.Word2Vec.load('./ja/ja.bin')

		### feature作成 ####
		df = getGeneralityInfo(df, userID_1902)		#dim1
		df = getContextInfo(df)						#dim1
		df = getIfQuestion(df)						#dim1
		df = getDependNodWord(df)					#dim2
		df = getDialogueAct(df)						#dim5

		dfs = []
		for ID in userID_1902:
			print(ID)
			df_cnt = df[df['name'].str.startswith(ID)]
			#### feature作成 ####
			df_cnt = getUserImpression(df_cnt)		#dim2
			df_cnt = getUserUtteLength(df_cnt, ID)	#dim2
			df_cnt = getNumNoun(df_cnt)				#dim2
			df_cnt = getUtteranceThemeIndex(df_cnt)	#dim1

			dfs.append(df_cnt)
		df = pd.concat(dfs)

		# 使用しないシステム発話をあらかじめ取り除く
		# そのためにテーマ情報のファイルとmergeする
		THEMEdf = pd.read_csv('./1902themeInfo/1902MMcorpus_theme.csv')
		df = pd.merge(df, THEMEdf, on='agent_utterance')

		# 特徴量選択
		Xdata = df[fea_name].iloc[:, :].values
		if normalization == 'stand':# 標準化
			Xdata = preprocessing.scale(Xdata)
			for i in range(len(Xdata[0])):
				df[fea_name[i]+'_s'] = Xdata[:, i]
		elif normalization == 'norm':# 正規化
			Xdata = preprocessing.minmax_scale(Xdata)
			for i in range(len(Xdata[0])):
				df[fea_name[i]+'_n'] = Xdata[:, i]
		else:
			pass

		# クラスタリング，ファイル作成
		if cluster_way == 'KM':
			clsInfo = KMeans(Xdata, n_clusters=clsN)
			df['cls'] = clsInfo
			df.to_csv(params.get('path_main_class_info'), index=None)
		elif cluster_way == 'NH':
			clsInfo = RuleBaseClustering(Xdata, fea_name)
			df['cls'] = clsInfo
			df = df[['name','agent_utterance','cls']]
			df.to_csv(params.get('path_main_class_info'), index=None)
		elif cluster_way == 'KM2':
			n_clusters1 = int(input('1step cls num >> '))
			n_clusters2 = int(input('2step cls num >> '))
			clsInfo = KMeans_2step(Xdata, n_clusters1=n_clusters1, n_clusters2=n_clusters2)
			df['cls'] = clsInfo
			df.to_csv(params.get('path_main_class_info'), index=None)
		else:
			print('input invalid')
			exit(0)




	# 必要なファイルをまとめて作成
	if action == 'makefile':

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


		print('#####分けられたクラスごとにシステム発話を書き下し#####')
		CLSdf = pd.read_csv(params.get('path_main_class_info'))
		meta = ['cls', 'freq', 'freq_1cls', 'ratio', 'agent_utterance']
		SUdf = pd.DataFrame(data=[], columns=meta)
		clas = list(set(CLSdf['cls'].values))
		num = 0

		# 頻度の辞書
		freqDict = collections.Counter(CLSdf['agent_utterance'].values)

		for c in clas:
			df_1cls = CLSdf[CLSdf['cls'] == c]
			agent_utterance_1cls = list(set(df_1cls['agent_utterance'].values))
			freqDict_1cls = collections.Counter(df_1cls['agent_utterance'].values)

			for val in agent_utterance_1cls:
				SUdf.loc[num] = [c, freqDict[val], freqDict_1cls[val], round(float(freqDict_1cls[val])/freqDict[val], 2), val]
				num += 1

		THEMEdf = pd.read_csv('./1902themeInfo/1902MMcorpus_theme.csv')
		THEMEdf = THEMEdf.drop('freq', axis=1)
		SUdf = pd.merge(SUdf, THEMEdf, on='agent_utterance')

		df_sorted = SUdf.sort_values(['cls', 'theme'])
		df_sorted.to_csv(params.get('path_utterance_by_class'), index=None)

		print('*クラス内発話数*')
		c = collections.Counter(df_sorted['cls'].values)
		c_sorted = sorted(c.items(), key=lambda x:x[0])
		for i, val in enumerate(c_sorted):
			print(val[0],'\t',val[1])

		print('#####クラスタ内テーマごとの発話カウント#####')
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

		CLS_THEME_df = pd.DataFrame(data=info, columns=theme)
		CLS_THEME_df.to_csv(params.get('path_freq_class_theme'), index=None)

	# 対話行為8で発話クラス生成
	if action == 'clustering_da8':
		Uttedf = pd.read_csv('./UI3_su_da.csv')
		dialogFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/dialogue.csv')
		textFeadf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/text.csv')
		Uttedf = Uttedf[['name','agent_utterance','dialogue_act']]
		df = pd.merge(dialogFeadf[['name', 'lenS', 'label']], textFeadf[['name', '名詞']], on='name')
		df = pd.merge(df, Uttedf, on='name')
		df = remDataFrameError(df, meta='label', remove=True, devision=True)

		dfs = []
		for ID in userID_1902:
			print(ID)
			df_cnt = df[df['name'].str.startswith(ID)]
			#### feature作成 ####
			df_cnt = getUserImpression(df_cnt)		#dim2
			df_cnt = getUserUtteLength(df_cnt, ID)		#dim2
			df_cnt = getNumNoun(df_cnt)				#dim2

			dfs.append(df_cnt)
		df = pd.concat(dfs)

		# 使用しないシステム発話をあらかじめ取り除く
		# そのためにテーマ情報のファイルとmergeする
		THEMEdf = pd.read_csv('./1902themeInfo/1902MMcorpus_theme.csv')
		df = pd.merge(df, THEMEdf, on='agent_utterance')
		df = df.rename(columns={'dialogue_act': 'cls'})

		df.to_csv(params.get('path_main_class_info'), index=None)


	# 対話行為5で発話クラス生成
	if action == 'clustering_da5':
		basedf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/191024_fea15_norm.csv')
		CLSdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/da5.csv')

		utte = basedf['agent_utterance'].values
		da_cls = [''] * len(utte)
		for i, val in enumerate(utte):
			da_cls[i] = CLSdf[CLSdf['agent_utterance'] == val]['da5_new'].values[0]

		basedf['cls'] = da_cls
		basedf.to_csv(params.get('path_main_class_info'), index=None)




