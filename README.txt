README

ver2.0
ver1系はInteの推定，追跡するものだったが，
ver2,ver3系はUI3を推定する．

ver1系は実際にリアルタイムで対話出来るシステムだったが，
ver2では，対話用のコードは省略してPOMDPのモデル設計周りに関するコードを管理する．
ver3では，発話をクラスタリングをして，それを用いて対話システムを作成しています．


【各ファイルの説明】


XXX_average.csv
XXX_bycls.csv
XXX_freq_cls-theme.csv
XXX.csv
priprob_XXX.npy
STP_XXX.npy
対話に必要なファイル一式

XXXについて
    191015_da
    初期に設計した対話行為(8)を用いたモデル

    191024_fea15_norm
    KMeansでクラスタリングした提案手法，特徴量は15，特徴量を正規化

    191101_da5
    今回人手で作成した対話行為(5)を用いたモデル


190704_takedaSTP/
武田先生にSTPを変更してもらったっときのデータ

    190704baseSTP.csv
    これまでのベース手法でのU3推定結果

    190704takedaSTP.csv
    武田先生のSTPでの推定結果

    takedaT_STP.npy
    そのSTPデータ

    trans_all.csv
    UI3_trans.csv
    STPテンソル作成のために使用した統計データ


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


da5.csv
対話行為(5)の詳細

UI3_su_da.csv
対話行為(8)の詳細

191004_mix分析
分析たくさん

1902themeInfo
発話ごとのテーマ情報

clustering_sysutte
手法ごとの実験結果やファイル入れ

log
手法ごとの対話log入れ

登場する品詞の極性
ボツになったもの．極性を取ろうとした

分析ファイル
メモ.txt
まんま

contextInfo.txt
contextInfo2.txt
発話の組の前後関係を示したもの．bi-gram的な情報

README.txt
これ







