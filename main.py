# -*- coding:utf-8 -*-
import socket
import time
import subprocess
import extFaceFea as eff
import pandas as pd
import predFunc as pf
import extTextFea as etf
import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import threading
import datetime

from optparse import OptionParser
import controlFST as cf

####### 使用方法 ###############
# 音声認識をするために、C:\Users\kaldi\dictation-kit-v4.4\ASR.pyを実行してください。
# 別ターミナルで、C:\Users\kaldi\Desktop\main\main.pyを実行してください。
# 
# 以下の3つが連動して動いています．
# １．このコードの入ったディレクトリ(C:\Users\kaldi\Desktop\main)
# ２．julus音声認識のためのディレクトリ(C:\Users\kaldi\dictation-kit-v4.4)
# ３．MMDAgentで対話管理を行うためのディレクトリ(C:\Users\kaldi\Desktop\MMDAgent)
#
# 発話後、エンターキーを押してください。それが発話終了の合図になります。
# 音声認識が完了するまで少し時間がかかります。
# ASR.pyを実行した方のターミナルを確認し、音声認識が完了したことを確かめてからエンターキーを押してください。
###############################

class turnIdx:
	def __init__(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def cntup(self, action):
		if action == "hold":
			self.idx_hold += 1
		elif action == "dig":
			self.idx_dig += 1
		elif action == "change":
			self.idx_change += 1

	def reset(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def continuity(self):
		self.cont_num += 1

class interest:
	def __init__(self, queue_size=3):
		self.prev_interest = [''] * queue_size
		self.cnt_interest = 0

	def update(self, inte):
		self.prev_interest.append(self.cnt_interest)
		del self.prev_interest[0]
		self.cnt_interest = inte

	def get_interest(self):
		return self.cnt_interest

	def get_prev_ave_interest(self):
		prev_val = self.prev_interest.append(self.cnt_interest)
		prev_val = [x for x in self.prev_interest if type(x) is not str]
		return np.mean(prev_val)


##参考URL https://qiita.com/nadechin/items/28fc8970d93dbf16e81b
if __name__ == '__main__':
	# options
	parser = OptionParser()
	parser.add_option("-C","--choiseFST", choices=["base","pom"],
						default='pom', help="type of FST way", dest="choiseFST")
	parser.add_option("-D","--maindata", choices=["online","offline"],
						default='online', help="type of main data type", dest="datatype")
	(options, args) = parser.parse_args()

	# feaのはいったファイルを入力としてオフライン推定
	if options.datatype == 'offline':
		# initialize
		state = np.array([[0.5], [0.5]])
		inte = interest()

		# 推定
		for i in range(50):
			df = pd.read_csv('./train_data/dialogue.csv')
			d_X_test = df.iloc[i, 1:-1].values
			model = pickle.load(open('./model/d.model', 'rb'))
			d_pred = pf.predUnknown(d_X_test, model, 'd')

			df = pd.read_csv('./train_data/voice.csv')
			df = pf.selectNbest(df)
			v_X_test = df.iloc[i, :].values
			model = pickle.load(open('./model/v.model', 'rb'))
			v_pred = pf.predUnknown(v_X_test, model, 'v')

			df = pd.read_csv('./train_data/text.csv')
			pca = pickle.load(open('./model/pca.model', 'rb'))
			df = pf.PCAonlyBOW(df, pca=pca)
			t_X_test = df.iloc[i, 1:-1].values
			model = pickle.load(open('./model/t.model', 'rb'))
			t_pred = pf.predUnknown(t_X_test, model, 't')

			df = pd.read_csv('./train_data/face.csv')
			f_X_test = df.iloc[i, 1:-1].values
			model = pickle.load(open('./model/f.model', 'rb'))
			f_pred = pf.predUnknown(f_X_test, model, 'f')

			###### fusion #######
			df = pd.DataFrame(data=[[d_pred, v_pred, t_pred, f_pred]])
			X_test = df.iloc[0, :].values
			model = pickle.load(open('./model/fusion.model', 'rb'))
			fusion_pred = pf.predUnknown(X_test, model, 'fusion')

			# 更新
			inte.update(fusion_pred)
			state = cf.updateStatePOMDP(state, inte, False)

			print(state)
			print('{}発話対経過しました'.format(i+1))


	if options.datatype == 'online':

		if options.choiseFST == 'pom':
			from controlFST import makeCommandMessage as MCM

		# commands
		cmd_run_mmda = 'C:/Users/kaldi/Desktop/MMDAgent/MMDAgent/MMDAgent.exe C:/Users/kaldi/Desktop/MMDAgent/MMDAgent/MMDAgent_Example.mdf'
		openface_filename = 'sample'
		cmd_run_openface = 'C:/Users/kaldi/Desktop/OpenFace_2.0.5_win_x64/FeatureExtraction.exe -device 0 -of C:/Users/kaldi/Desktop/OpenFace_2.0.5_win_x64/processed/{}'.format(openface_filename)
		cmd_run_opensmile = 'C:/Users/kaldi/Desktop/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe -C C:/Users/kaldi/Desktop/opensmile-2.3.0/config/IS09_emotion.conf -I ./speech_wav/{}.wav -O ./speech_wav/{}.arff'
		cmd_rec_voice = 'sox -c 1 -r 16k -b 16 -t waveaudio MMDAmic ./speech_wav/{}.wav'
		cmd_arfftocsv = "python arffToCsv.py"
		# path
		path_usr_utterance_info = 'C:/Users/kaldi/dictation-kit-v4.4/userUtteranceInfo.csv'
		path_usr_voice_feature = 'C:/Users/kaldi/Desktop/main/speech_wav/{}.csv'

		# run MMDAgent
		p1 = subprocess.Popen(cmd_run_mmda.split(" "))
		# run openface
		p2 = subprocess.Popen(cmd_run_openface.split(" "))
		# wait runnning
		time.sleep(5)
		# connect MMdagent server
		host, port = "localhost", 39390
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.connect((host, port))

		# initialize
		action = "hold"
		state = np.array([[0.5], [0.5]])
		inte = interest()
		theme = "sports"
		turnIdx = turnIdx()
		userID = 'nishimoto'
		reward = 0

		# basetime
		base_time = time.time()
		# 音声認識結果ファイルの既読情報
		already_read_idx = 0
		# log info
		dt_now = str(datetime.datetime.now())
		dt_now = dt_now.translate(str.maketrans({'-': '', ' ': '', ':': '', '.': ''}))
		f = open('C:./log/{}_{}_log.txt'.format(dt_now[:12], userID), 'w')
		f.write('record time : {}\n'.format(dt_now))
		with open('C:./refData/logMeta.txt', 'r')as metaf:
			meta = [x.replace("\n", "") for x in metaf.readlines()]
		f.write("\t".join(meta) + "\n")
		
		############## メインのループ ###################
		for i in range(20):
			print("対話ID : {} です。".format(i+1))
			# make filename "{ID}_{exchg_idx}"
			file_name = '{}_{}'.format(userID, str(i).zfill(3))
			# update state by POMDP
			state = cf.updateStatePOMDP(state, inte, False)
			print('state :')
			print(state)
			# send msg(system utterance) to MMDAgent
			msg, state, action, turnIdx = MCM(action, state, inte, theme, turnIdx)
			client.send(msg)
			# print sys start time
			s_s = time.time() - base_time
			print('sys_start : {:.4}'.format(s_s))
			# recv MMDAgent msg
			while True:
				response = client.recv(4096)
				if 'SYNTH_START' in response.decode('shift_jis'):
					recvInfo = response.decode('shift_jis').split('\r\n')
					for i, val in enumerate(recvInfo):
						if 'SYNTH_START' in val:
							s_len = len(recvInfo[i].split('|')[-2])
							da = recvInfo[i].split('|')[-1].replace('\r\n', '')

				if 'SYNTH_EVENT_STOP' in response.decode('shift_jis'):
					s_e = time.time() - base_time
					print('sys_end : {:.4}'.format(s_e))
					break

				if 'THEME_CHANGE' in response.decode('shift_jis'):
					recvInfo = response.decode('shift_jis').split('\r\n')
					for i, val in enumerate(recvInfo):
						if 'THEME_CHANGE' in val:
							theme = recvInfo[i].split('|')[1]
					turnIdx.reset()
					state = np.array([[0.5], [0.5]])
			# rec voice by sox
			cmd_rec_voice_cnt = cmd_rec_voice.format(file_name)
			p3 = subprocess.Popen(cmd_rec_voice_cnt.split(" "))
			# ユーザ発話終了をユーザのエンターキー押下で取得	
			while True:
				input("Push Enter key if utterance finished...")
				u_e = time.time() - base_time
				print('u_end : {:.4}'.format(u_e))
				p3.terminate()
				break
			# run opensmile
			cmd_run_opensmile_cnt = cmd_run_opensmile.format(file_name, file_name)
			p4 = subprocess.call(cmd_run_opensmile_cnt.split(" "))
			# change arff -> csv
			p5 = subprocess.call(cmd_arfftocsv.split(" "))

			####### ext utterance ######
			df = pd.read_csv(path_usr_utterance_info)
			utterance = ''.join(df['word'].values[already_read_idx:].tolist())
			u_s = df['u_s'].values[already_read_idx] - base_time
			print('u_start : {}'.format(u_s))
			print('utterance -> {}'.format(utterance))
			already_read_idx = len(df)

			###### predict mono modal ######
			d_pred, v_pred, t_pred, f_pred = None, None, None, None
		
			def dialogue():
				global d_pred
				u_len = len(utterance)
				df = pf.makeDiaDF(u_s - s_e, s_len, len(utterance), s_len - u_len, da)
				X_test = df.iloc[0, :].values
				model = pickle.load(open('./model/d.model', 'rb'))
				d_pred = pf.predUnknown(X_test, model, 'd')

			def voice():
				global v_pred
				df = pd.read_csv(path_usr_voice_feature.format(file_name))
				df = pf.selectNbest(df)
				X_test = df.iloc[0, :].values
				model = pickle.load(open('./model/v.model', 'rb'))
				v_pred = pf.predUnknown(X_test, model, 'v')

			def text():
				global t_pred
				df = etf.makeFea(utterance)
				pca = pickle.load(open('./model/pca.model', 'rb'))
				df = pf.PCAonlyBOW(df, pca=pca)
				X_test = df.iloc[0, 1:-1].values
				X_test = X_test.astype(np.float32)
				model = pickle.load(open('./model/t.model', 'rb'))
				t_pred = pf.predUnknown(X_test, model, 't')

			def face():
				global f_pred
				X_test = eff.predictionFace(s_s, u_e)
				model = pickle.load(open('./model/f.model', 'rb'))
				f_pred = pf.predUnknown(X_test, model, 'f')

			t1 = threading.Thread(target=dialogue)
			t2 = threading.Thread(target=voice)
			t3 = threading.Thread(target=text)
			t4 = threading.Thread(target=face)
			t1.start()
			t2.start()
			t3.start()
			t4.start()

			thread_list = threading.enumerate()
			thread_list.remove(threading.main_thread())
			for thread in thread_list:
				thread.join()

			###### fusion 4 modal #######
			df = pd.DataFrame(data=[[d_pred, v_pred, t_pred, f_pred]])
			X_test = df.iloc[0, :].values
			model = pickle.load(open('./model/fusion.model', 'rb'))
			fusion_pred = pf.predUnknown(X_test, model, 'fusion')
			inte.update(fusion_pred)

			print('####################################################')

			f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(cf.sigmoid(d_pred),
				cf.sigmoid(v_pred),cf.sigmoid(t_pred),cf.sigmoid(f_pred),cf.sigmoid(fusion_pred)))

		client.close()
		p1.terminate()
		p2.terminate()



