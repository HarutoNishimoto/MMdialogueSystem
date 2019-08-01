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


# 入力をcsvとしてdataを作成
def readCsv(df):
	meta = df.columns.values
	index = df['name'].values
	Xdata = df.iloc[:, 1:-1].values
	label = df['label'].values
	return meta, index, Xdata, label

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



if __name__ == '__main__':


	# ここは関数の保持のみを行うコード
	print('it is the code has main function.')



