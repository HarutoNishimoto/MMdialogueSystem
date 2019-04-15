# -*-coding: utf-8 -*-

import re
import copy
import numpy as np
import sys
import os.path
import MeCab
import pandas as pd


# 1_base.mecab読み込んで整形してreturn
def read_base_file(user_id, path_to_np):

	# ステミングしたtxtファイル読み込み
	read_path = path_to_np + user_id + '_base.mecab'
	with open(read_path, 'r')as rf:
		base_lines = rf.readlines()

	# メインで使うdata（リストのリスト）
	base_data = []
	for i in range(len(base_lines)):
		# str->unicodeに変換（そうすることで日本語の文字数とかが扱いやすくなる）
		base_lines[i] = base_lines[i].decode('utf-8')
		base_data.append(re.split('[ ]', base_lines[i]))

	return base_data

# 1_pos.mecab読み込んで整形してreturn
def read_pos_file(user_id, path_to_np):
	# ステミングしたtxtファイル読み込み
	read_path = path_to_np + user_id + '_pos.mecab'
	with open(read_path, 'r')as rf:
		pos_lines = rf.readlines()

	# メインで使うdata（リストのリスト）
	pos_data = []
	for i in range(len(pos_lines)):
		# str->unicodeに変換（そうすることで日本語の文字数とかが扱いやすくなる）
		pos_lines[i] = pos_lines[i].decode('utf-8')
		pos_data.append(re.split('[ ]', pos_lines[i]))

	return pos_data


#search_wordがword_list（リスト）に存在するかを返す関数
def is_in_speech(word_list,search_word):
	for word in word_list:
		if search_word == word:
			return True

	return False
		

# 独自素性
# 文章から素性を抽出し，そのベクトルを返す．
# 素性数5
def ext_origin(user_id, path_to_np, base_data=None, pos_data=None):

	if base_data == None:
		#ファイル読み込み
		base_data = read_base_file(user_id, path_to_np)
		pos_data = read_pos_file(user_id, path_to_np)

	# 素性の値:興味あり->1 興味なし->-1 不明->0
	text_features = []
	for line in base_data:
		text_features.append([])

	##fea1（無言なら興味ない）
	num = 0
	for line in base_data:
		if u'＊' in line:
			text_features[num].append(-1)
		else:
			text_features[num].append(0)
		num += 1

	##fea2-5（名詞，形容詞，副詞，感動詞の個数）
	num = 0
	for line in pos_data:
		text_features[num].append(line.count(u'名詞'))
		text_features[num].append(line.count(u'形容詞'))
		text_features[num].append(line.count(u'感動詞'))
		text_features[num].append(line.count(u'副詞'))
		num += 1

	return text_features

# bag−of−words（有無）
# 文章から素性を抽出し，そのベクトルを返す．
# 素性数（5または10のデータの総単語数）
def ext_bow(user_id, path_to_np, base_data=None):

	# 付属語削除，使用回数制限
	#all_wordlist = rem_huzokugo(path_to_np)
	all_wordlist = np.load('./refData/{}wordList_rem.npy'.format(path_to_np))

	text_features = []

	if base_data == None:
		#ファイル読み込み
		base_data = read_base_file(user_id, path_to_np)

	# text_featuresの作成
	for line in base_data:
		line_data = []
		for word in all_wordlist:
			if is_in_speech(line, word):
				line_data.append(1)
			else:
				line_data.append(0)
		text_features.append(copy.deepcopy(line_data))

	return text_features


# hybrid-BOW（有無）
# 文章から素性を抽出し，そのベクトルを返す．
# 素性数（「5または10のデータの総単語数」+「独自に作成したX個」）
def ext_hbow(user_id, path_to_np):

	# bow
	text_features_bow = ext_bow(user_id, path_to_np)
	# origin
	text_features_hand = ext_origin(user_id, path_to_np)

	# BOW，独自素性の長さの一致の確認
	if len(text_features_bow) == len(text_features_hand):
		#print 'comment:同じデータ数です．'

		text_features = []
		for i in range(len(text_features_bow)):
			a_text_features = []
			#early fusion
			for feature in text_features_bow[i]:
				a_text_features.append(feature)
			for feature in text_features_hand[i]:
				a_text_features.append(feature)
			text_features.append(a_text_features)
	else:
		pass
		#print 'error:異なるデータ数です．'
	return text_features


def makeTmeta():
	#省略
	#wordlist = rem_huzokugo()
	wordlist = np.load('./refData/wordList_rem.npy')

	bow = ['word#'] * len(wordlist)
	for i, val in enumerate(bow):
		bow[i] = bow[i] + str(i+1).zfill(3)
	origin = ['無言', '名詞', '形容詞', '感動詞', '副詞']
	meta = ['name'] + bow + origin + ['label']
	return meta


# 入力を1文としてfea抽出
def makeFea(input_text):

	m = MeCab.Tagger()
	morphs = m.parse(input_text)

	morphs = morphs.split('\n')
	for i, morph in enumerate(morphs):
		morphs[i] = morphs[i].split(',')

	pos_data, base_data = [], []
	pos, base = '', ''
	for i, val in enumerate(morphs):
		if val[0] != 'EOS':
			pos = pos + str(val[0].split('\t')[1]) + ' '
			base = base + str(val[6]) + ' '
		else:
			pos_data.append(pos)
			base_data.append(base)
			break
	########### ここコメントアウトしてるけど、なぜかは不明 ################
	#for i, val in enumerate(pos_data):
	#	pos_data[i] = pos_data[i].decode('utf-8')
	#	base_data[i] = base_data[i].decode('utf-8')

	#################無言の処理は??????????ーーーーーーーー

	path_to_np = 'C:/Users/kaldi/Desktop/main/'
	bow_fea = ext_bow(0, path_to_np, base_data=base_data)
	org_fea = ext_origin(0, path_to_np, base_data=base_data, pos_data=pos_data)

	# 未知のデータなので，nameとlabelはなんでもいい
	dammy = [['?']]
	text_features = np.hstack((dammy, bow_fea, org_fea, dammy))
	meta = makeTmeta()
	df = pd.DataFrame(data=text_features, columns=meta)
	return df

# allwordlist.npyを読み込み，付属語削除と使用頻度制限してlistをreturn
def rem_huzokugo(path_np, f_name='wordList', freq=1):
	# 読み込み
	wordList = np.load('{}{}.npy'.format(path_np, f_name)).tolist()
	print('before_rm:{}'.format(len(wordList)))

	#登場回数freq回以下の人は削除
	wordList = [e for e in set(wordList) if wordList.count(e) > freq]

	model = MeCab.Tagger()
	for word in wordList:
		info = model.parse(word.encode('utf-8'))
		info = re.split('[\t,]', info)
		if len(info) > 1:
			pos = info[1]
			if pos == '助詞' or pos == '助動詞':
				wordList.remove(word)
	# 「\n」は削除
	wordList.remove(u'\n')
	# ソート
	wordList.sort()

	print('after_rm:{}'.format(len(wordList)))

	return wordList

if __name__ == '__main__':

	print('this is code for ext text fea')
