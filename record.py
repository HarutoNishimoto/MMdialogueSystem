# -*-coding: utf-8 -*-
import subprocess
import sys
import time

# 実行したらwav作成します．
# エンターを押すまでの間，録音されます．
# wavのファイル名をsys[1]で入力する
if __name__ == '__main__':

	now_time = time.time()
	print('start time -> {}'.format(now_time))
	
	print("Start recording...")
	p = subprocess.Popen("sox -c 1 -r 16k -b 16 -t waveaudio MMDAmic ./speech_wav/{}.wav".format(sys.argv[1], shell=True))

	input('')
	p.terminate()


