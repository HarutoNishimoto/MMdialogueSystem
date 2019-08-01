README

ver2.0
ver1系はInteの推定，追跡するものだったが，
ver2系はUI3を推定する．

ver1系は実際にリアルタイムで対話出来るシステムだったが，
ver2では，対話用のコードは省略してPOMDPのモデル設計周りに関するコードを管理する．

【各ファイルの説明】
    190618_pred_state.csv
    いつのかわからないUI3の推定結果

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

    offline.py
    基本的にメインの活動場所．
    POMDPで推定したり，以前のものと比較したり．

    controlFST.py
    POMDPないのSTPやpolicyやパラメータの管理

    main.py
    classの保管場所

    makeTransFunc.py
    STPや事前確率をデータから計算

    predFunc.py
    値の変換，データの読み込み（あまり使用しないっぽい感じがする）

    param_priprob_UI3_withA.npy
    param_STP_UI3_withA.npy
    事前確率，STPのデータが入ったnumpyテンソル

    parameters.txt
    パラメータ

    pred_UI3_POMDP_alpha-04_sigma-04.csv
    現時点でグリッドサーチで最高の精度がでたモデル
    alphaは尤度関数の分母の乗数
    sigmaは連続値計算のための正規分布の分散

    README.txt
    これ
