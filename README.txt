README

ver2.0
ver1系はInteの推定，追跡するものだったが，
ver2,ver3系はUI3を推定する．

ver1系は実際にリアルタイムで対話出来るシステムだったが，
ver2では，対話用のコードは省略してPOMDPのモデル設計周りに関するコードを管理する．
ver3では，発話をクラスタリングをして，それを用いて対話システムを作成しています．

parameters.txt
これのpathやdefineClass内のこれを指すpathを適切にすれば
発話クラスタリングをした上での対話デモはできる．（発話クラスタ選択手法は「dim3」）


【各ファイルの説明】

XXX_average.csv
XXX_bycls.csv
XXX_freq_cls-theme.csv
XXX.csv
priprob_XXX.npy
STP_XXX.npy
対話に必要なファイル一式

XXXについて
    191024_fea15_norm
    KMeansでクラスタリングした提案手法，特徴量は15，特徴量を正規化
    一旦SLUD2019までではこれが一番いいことにしている


controlBelief.py
POMのpolicyなどを管理する場所

defineClass.py
クラスを定義している場所

makeBeliefFunc.py
STPやpriprobを作成する

predOffline.py
メインで動かすもの．システム対話をする

sys_utterance.py
システム発話の特徴量抽出やクラスタリングをしてファイル作成する

analysys.py
分析のためのコードいろいろ

old_dialogue_act
これまで作成した対話行為の保管場所

191004_mix分析
分析たくさん

1902themeInfo
発話ごとのテーマ情報

clustering_sysutte
手法ごとの実験結果やファイル入れ

log
手法ごとの対話log入れ

分析ファイル
メモ.txt
まんま

README.txt
これ

requirements.txt
condaでpy36の環境を作成．その後に必要なパッケージ

parameters.txt
ファイルのパス管理



