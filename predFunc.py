# -*-coding: utf-8 -*-

import numpy as np
import sys
from sklearn import preprocessing
from sklearn import svm
import pandas as pd
import copy
from sklearn.decomposition import PCA
import pickle
import controlFST as fst

# 桁数指定
np.set_printoptions(precision=3)

# voiceの特徴量選択
def selectNbest(df):
	# ファイルから使用するfea指定
	with open('./refData/voice_fea10.txt', 'r')as rf:
		useVFea = rf.readlines()
	useVFea = [x.replace('\n', '') for x in useVFea]
	df_new = df[useVFea]
	return df_new

# dialogueの素性df作成
def makeDiaDF(reaction, s_len, u_len, su_len, da):
	da_list = ['io','na','oa','op','pa','qw','qy','su']
	idx = da_list.index(da)
	da_fea = np.eye(8)[idx]
	fea = [reaction, s_len, u_len, su_len] + da_fea.tolist()
	df = pd.DataFrame(data=[fea],
		columns=['reaction','lenS','lenU','lendiff','io','na','oa','op','pa','qw','qy','su'])
	return df

# 入力をcsvとしてdataを作成
def readCsv(df):
	meta = df.columns.values
	index = df['name'].values
	Xdata = df.iloc[:, 1:-1].values
	label = df['label'].values

	return meta, index, Xdata, label

# X_testは1データ分の特徴量ベクトル
# 1次元でnumpy形式指定で．
def predUnknown(X_test, model, fea_type):
	X_test = np.array([X_test.tolist()])
	# 説明変数を取り出した上でスケーリング
	scale = np.load('C:/Users/kaldi/Desktop/main/model/{}_scale.npy'.format(fea_type))
	maxi = scale[0]
	mini = scale[1]
	X_test[0] = (X_test[0] - mini) / (maxi - mini)
	# 推定
	y_pred = model.predict(X_test)

	print(y_pred)

	print('{}の推定結果 : {} / o_prob : {:.3}'.format(fea_type, y_pred, fst.sigmoid(y_pred[0])))
	return y_pred[0]


# 分類器作成
def makeModel(df, clf, path, fea_type='v'):
	# 実数値に変換
	label = label2float(df['label'].values)
	df['label'] = label
	meta, index, X_train, y_train = readCsv(df)
	# 説明変数を取り出した上でスケーリング
	X_train = preprocessing.minmax_scale(X_train)
	# 分類器の構築
	clf.fit(X_train, y_train)
	# save
	pickle.dump(clf, open('{}{}.model'.format(path, fea_type), 'wb'))


# textPCA(bow)のモデル作成
def makePCAModel(df, path, pca_dim=5):
	meta, index, Xdata, label = readCsv(df)

	word_num = len([x for x in meta if 'word#' in x])
	XdataBow = Xdata[:, :word_num]
	XdataOrg = Xdata[:, word_num:]

	pca = PCA(n_components=pca_dim, random_state=0)
	pca.fit(XdataBow)
	pickle.dump(pca, open('{}pca.model'.format(path), 'wb'))

# oxを+-1に変更してreturn
def label2float(label_list):
	new_label_list = [''] * len(label_list)
	for i,val in enumerate(label_list):
		if val == 'o':
			new_label_list[i] = +1
		elif val == 'x':
			new_label_list[i] = -1
	return new_label_list

# +-1をoxに変更してreturn
def float2label(label_list, thres=0):
	new_label_list = [''] * len(label_list)
	for i,val in enumerate(label_list):
		if val >= thres:
			new_label_list[i] = 'o'
		elif val < thres:
			new_label_list[i] = 'x'
	return new_label_list

# 入力ベクトル
# 出力min0max1に線形変換したベクトル
def min_max(x, axis=None):
	min = x.min(axis=axis, keepdims=True)
	max = x.max(axis=axis, keepdims=True)
	if max-min == 0:
		return np.zeros(len(x))
	else:
		result = (x-min)/(max-min)
		return result

def KBest(df, fea_num=5):
	from sklearn.feature_selection import SelectKBest, f_regression

	meta, index, Xdata, label = readCsv(df)
	label = label2float(label)
	selector = SelectKBest(score_func=f_regression, k=fea_num) 
	selector.fit(Xdata, label)
	mask = selector.get_support()

	select_meta = [m for m, msk in zip(meta[1:-1], mask) if msk]
	select_meta = ['name'] + select_meta + ['label']

	X_selected = selector.transform(Xdata)
	label = float2label(label)
	label = np.array([label]).transpose()
	index = np.array([index]).transpose()
	data = np.hstack((index, X_selected, label))
	df = pd.DataFrame(data=data, columns=select_meta)
	return df

def PCAonlyBOW(df, pca_dim=5, pca=None):                 
	meta, index, Xdata, label = readCsv(df)
	word_num = len([x for x in meta if 'word#' in x])
	XdataBow = Xdata[:, :word_num]
	XdataOrg = Xdata[:, word_num:]

	if pca == None:
		pca = PCA(n_components=pca_dim, random_state=0)
		pca.fit(XdataBow)

	XdataBow = pca.transform(XdataBow)
	new_Xdata = np.hstack((XdataBow, XdataOrg))

	## メタ作成
	meta = ['t_fea#'] * len(new_Xdata[0])
	for i, val in enumerate(meta):
		meta[i] = meta[i] + str(i+1).zfill(2)
	meta = ['name'] + meta + ['label']

	new_arr = np.hstack((np.array([index]).transpose(), new_Xdata, np.array([label]).transpose()))
	df = pd.DataFrame(data=new_arr, columns=meta)

	return df

# 特徴量スケーリングのmaxminの書き出し
def scaling(arrX, path, fea_type):
	maxi = arrX.max(axis=0).tolist()
	mini = arrX.min(axis=0).tolist()
	max_min = np.array([maxi, mini])
	np.save('{}{}_scale.npy'.format(path, fea_type), max_min)
	return 0




if __name__ == '__main__':


	# ここは関数の保持のみを行うコード
	print('it is the code has main function.')



