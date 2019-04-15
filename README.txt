README

【対話システムの起動方法】
    1. ターミナルを2つ起動させます．（音声認識用と対話管理用）
    2. 音声認識用のターミナルは'activate py27'でpython2の環境にします．
    3. 対話管理用のターミナルは'activate py36'でpython3の環境にします．
    4. 音声認識のものは\Users\kaldi\dictation-kit-v4.4\に移動して，'python ASR_py2.py'で実行．（15秒程度待機が必要です．）
    5. 対話管理のものは\Users\kaldi\Desktop\main\に移動して，'python main.py'で実行．

    〈対話中の注意〉
    自分の発話終了した場合，エンターキーを押下してください．
    コードの都合上，何かしらの音声認識結果がないと対話が続かないです．
    なので，音声認識用のターミナルに音声認識結果が表示されてからキーを押すようにしてください．

    〈対話終了後の注意〉
    対話を終了したい場合，どちらのターミナルも'ctrl + C'で終了させてください．
    （顔の表情をとるopenfaceなどはそのウィンドウのｘを押しても消えません．）

    〈その他〉
    パラメータは適宜変更してください．

    〈TODO〉
    使用用途に応じたコードになっていればいいね．
    現状SVRで推定しているから，LRに変更しなくてはならない
    モデルとか統合推定とかをいろいろ変更しなくてはならない
    人に見せる用に，うまくいく条件とかをあらかじめ覚えておく
    あくまでメイちゃんと話している感が出るようにしたいので，ウィンドウの表示を考える．


【各ファイルの説明】
    model
    winで作成したmodel

    train_data
    alldata2578のcsv

    speech_wav
    録音した音声ファイルの格納場所

    actionunit.txt
    userID.csv
    voice_fea10.txt
    いるねん

    mainFunction.py
    モジュール群

    makeModel.py
    モデル作成用

    openface.py
    リアルタイムface

    extTextFea.py
    text特徴量抽出

    main.py
    これで全部できる

    controlFST.py
    FST管理。命令や行動のコマンドを送信する

    arffToCsv.py
    毎度おなじみコンバーター

    record.py
    soxで録音するためのもの


【\Users\kaldi\dictation-kit-v4.4\にはいっているもの】
    ASR.py
    リアルタイム音声認識
    juliusってdictationkitだけあればいいっぽい

    tsInfo.csv
    ASR結果のtsとwordの格納場所

