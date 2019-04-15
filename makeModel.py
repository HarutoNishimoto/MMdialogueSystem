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
	d_path = 'C:/Users/kaldi/Desktop/main/train_data/dialogue.csv'
	v_path = 'C:/Users/kaldi/Desktop/main/train_data/voice.csv'
	t_path = 'C:/Users/kaldi/Desktop/main/train_data/text.csv'
	f_path = 'C:/Users/kaldi/Desktop/main/train_data/face.csv'
	fusion_path = 'C:/Users/kaldi/Desktop/main/train_data/fusion.csv'
	# clfの指定
	clf = svm.SVR(kernel='rbf')
	# userID
	df = pd.read_csv('C:/Users/kaldi/Desktop/main/refData/userID.csv', header=None)
	userID = df[0].values

	# model作成
	# maxminのスケールもnpyで保存
	print('input:d/v/t/f/fusion/pca')
	fea_type = input('>>>  ')

	if fea_type == 'pca':
		df = pd.read_csv(t_path)
		pf.makePCAModel(df, 'C:/Users/kaldi/Desktop/main/model/')
	else:
		if fea_type == 'd':
			df = pd.read_csv(d_path)
		elif fea_type == 'v':
			df = pd.read_csv(v_path)
			df = pf.KBest(df, fea_num=10)
		elif fea_type == 't':
			df = pd.read_csv(t_path)
			df = pf.PCAonlyBOW(df)
		elif fea_type == 'f':
			df = pd.read_csv(f_path)
		elif fea_type == 'fusion':
			df = pd.read_csv(fusion_path)	

		model_path = 'C:/Users/kaldi/Desktop/main/model/'
		# 分類器の作成
		pf.makeModel(df, clf, model_path, fea_type=fea_type)
		# maxminのスケーリングの作成
		pf.scaling(df.iloc[:, 1:-1], model_path, fea_type)