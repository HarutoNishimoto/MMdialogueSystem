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

	# コーパス内の発話の使用頻度をカウントして，0回のものは削除
	# 1回しか使用しません
	if options.action == 'cntusefreq':
		THEMEdf = pd.read_csv(params.get('path_theme_info'))
		UTTEdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/UI3_su_da.csv')
		utte_freq = collections.Counter(UTTEdf['agent_utterance'].values)

		use_utte = THEMEdf['agent_utterance'].values
		use_freq = []
		for i, val in enumerate(use_utte):
			use_freq.append(utte_freq[val])
		THEMEdf['freq'] = use_freq
		THEMEdf = THEMEdf[['theme','freq','agent_utterance']]
		THEMEdf = THEMEdf[THEMEdf['freq'] != 0]	# 使用されなかったものは取り除く
		THEMEdf.to_csv(params.get('path_theme_info').split('.')[0] + '2.csv', index=None)

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




	# クラスごとの発話数
	if options.action == 'index':
		CLSdf = pd.read_csv('190926_fea8_clsInfo.csv')








