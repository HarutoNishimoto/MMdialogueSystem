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
	df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/refData/userID_1902.txt', header=None)
	userID_1902 = df[0].values
	# param get
	params = defineClass.params()


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


	# TCラベルを表示
	if options.action == 'TClabel':
		TCdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/annotate/TCanno/1902F3002.csv')
		TCaverage = TCdf[['F1','F2','F3','F5','M2']].values
		val = np.average(TCaverage, axis=1)
		print(val)
		plt.plot(range(len(val)), val)
		plt.show()



	# mecabが取れる名詞を確認
	if action == 'getnoun':
		mt = MeCab.Tagger()

		for ID in userID_1902:
			df1 = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_U.txt'.format(ID), header=None)
			df2 = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/speech/{}_S.txt'.format(ID), header=None)
			utte1 = df1[0].values
			utte2 = df1[0].values
			utte = utte1 + utte2

			for i, val in enumerate(utte):
				node = mt.parseToNode(val)
				while node:
					fields = node.feature.split(",")
					#print(fields)
					# 名詞、動詞、形容詞に限定
					if fields[0] == '名詞':
						print(node.surface, fields[1], fields[2], fields[3])
					node = node.next








	# クラスに名前つけ
	if options.action == 'naming':
		df = pd.read_csv(params.get('path_utterance_by_class'))
		NAMEdf = pd.read_csv(params.get('path_class_name'))
		newCLS = [''] * len(df)
		for i, c in enumerate(df['cls'].values):
			newCLS[i] = NAMEdf[NAMEdf['clsKM'] == c]['clsname'].values[0]

		df = df.rename(columns={'cls': 'clsKM'})
		df['cls'] = newCLS
		df.to_csv(params.get('path_utterance_by_class').split('.')[0] + '_named.csv', index=None)






	# クラスに名前つけ
	if options.action == 'sort':
		df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/RL/exchgUI3Info.csv')
		s_df = df.sort_values(['sys_utterance', 'user_utterance', 'UI3average'])
		s_df.to_csv('exchgUI3Info_sort.csv', index=None)







