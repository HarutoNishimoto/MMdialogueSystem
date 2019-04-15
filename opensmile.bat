@echo off

rem コマンドライン引数の1番目にファイル名を指定

C:/Users/kaldi/Desktop/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe ^
-C C:/Users/kaldi/Desktop/opensmile-2.3.0/config/IS09_emotion.conf ^
-I ./speech_wav/%1.wav -O ./speech_wav/%1.arff
