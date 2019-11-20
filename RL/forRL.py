# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
from optparse import OptionParser
import collections
from sklearn import preprocessing
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import random



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
	df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values



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


















