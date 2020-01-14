README

ver2.0
ver1系はInteの推定，追跡するものだったが，
ver2以降はUI3を推定する．

ver1系は実際にリアルタイムで対話出来るシステムだったが，
ver2では，対話用のコードは省略してPOMDPのモデル設計周りに関するコードを管理する．
ver3では，発話をクラスタリングをして，それを用いて対話システムを作成しています．
ver4では，強化学習を用いた発話選択をしています．

parameters.txt
これのpathやdefineClass内のこれを指すpathを適切にすれば
発話クラスタリングをした上での対話デモはできる．（発話クラスタ選択手法は「dim3」）


【各ファイルの説明】

XXX_average.csv
XXX_bycls.csv
XXX_freq_cls-theme.csv
XXX.csv
対話に必要なファイル一式

XXXについて
    200109_cls8
    これが一番いい対話行為セットとします（2020/01/14）


RL/
	dialogue_env.py
	el_agent.py
	heatmapQ.py
	q_learning.py
	virtual_user.py
	必須のコード

	forRL.py
	makeDA.py
	対話行為を作成したり，ファイル整理をするコード

	simpleDA.csv
	使用する発話の簡単な対話行為（手で作成した）

	学習結果
	まんま

	対話行為の比較.txt
	いい対話行為セットを作成するためにいろいろ考えたことがまとめられている

	requirements.txt
	環境名（rl−book）のモジュール．
	python=3.6.7


getSysUtteranceFea.py
システム発話の特徴量を取ってくる

controlBelief.py
POMのpolicyなどを管理する場所

defineClass.py
クラスを定義している場所

makeBeliefFunc.py
STPやpriprobを作成する

predOffline.py
特徴量のユークリッド距離を用いてシステム対話をする（ver3までのもの）

sys_utterance.py
システム発話の特徴量抽出やクラスタリングをしてファイル作成する

analysys.py
分析のためのコードいろいろ

old_dialogue_act
これまで作成した対話行為の保管場所

1902themeInfo
発話ごとのテーマ情報

clustering_sysutte
手法ごとの実験結果やファイル入れ

refData
引用，参考にするファイル群

log
手法ごとの対話log入れ

分析ファイル
メモ.txt
まんま

parameters.txt
ファイルのパス管理

191024_fea15_norm_bycls_limited1912.csv
191024_fea15_norm_bycls_named191024.csv
SIGSLUD時点での対話行為セット

200108-9_fea7_clsN.xlsx
一番いいモデルを作成する時に用いた分析ファイル

200110_baseDA4.csv
提案手法を支えるベースモデルで使用する対話行為セット




