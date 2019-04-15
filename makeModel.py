# -*-coding: utf-8 -*-

import numpy as np
import sys
from sklearn import svm
import pandas as pd
import predFunc as pf
from sklearn.decomposition import PCA


if __name__ == '__main__':

	# 読み込み
	# sklearnのために欲しいのはXdataとlabelだけ
	# clfの指定
	clf = svm.SVR(kernel='rbf')
	# userID
	df = pd.read_csv('./refData/userID.csv', header=None)
	userID = df[0].values

	# model作成
	# maxminのスケールもnpyで保存
	print('input:d/v/t/f/fusion/pca')
	fea_type = input('>>>  ')

	if fea_type == 'pca':
		df = pd.read_csv('./train_data/text.csv')
		pf.makePCAModel(df, './model/')
	else:
		if fea_type == 'd':
			df = pd.read_csv('./train_data/dialogue.csv')
		elif fea_type == 'v':
			df = pd.read_csv('./train_data/voice.csv')
			df = pf.KBest(df, fea_num=10)
		elif fea_type == 't':
			df = pd.read_csv('./train_data/text.csv')
			df = pf.PCAonlyBOW(df)
		elif fea_type == 'f':
			df = pd.read_csv('./train_data/face.csv')
		elif fea_type == 'fusion':
			df = pd.read_csv('./train_data/fusion.csv')	

		model_path = './model/'
		# 分類器の作成
		pf.makeModel(df, clf, model_path, fea_type=fea_type)
		# maxminのスケーリングの作成
		pf.scaling(df.iloc[:, 1:-1], model_path, fea_type)