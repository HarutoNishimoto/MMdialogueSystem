import pandas as pd
import numpy as np


#### ユーザモデルを定義するファイル

# ユーザモデル
class UserModel():

    def __init__(self):
        # 応答対を読み込み
        self.voca = pd.read_csv('exchgUI3Info.csv')
        self.UI = 4.0
        self.log_UI_1theme = []
        self.min_UI = 1.0
        self.max_UI = 7.0

    # システム発話に対して何か応答する
    # コーパスからs−uの交換の情報を用いて応答
    def getResponse(self, sys_utterance):
        # 候補が複数あるときはランダムに選択
        if '***' in sys_utterance:
            user_utterance = 'はい'
            self.UI = 4.0
            self.log_UI_1theme = [self.UI]
            UI_diff = 0.0
        else:
            responseInfo = self.voca[self.voca['sys_utterance'] == sys_utterance].sample()
            user_utterance = responseInfo['user_utterance'].values
            UI_diff = responseInfo['UI3average_diff'].values

            # UIが範囲[1, 7]を超えないように調整
            if self.UI + UI_diff < self.min_UI:
                self.UI = self.min_UI
            elif self.max_UI < self.UI + UI_diff:
                self.UI = self.max_UI
            else:
                self.UI += UI_diff
            self.log_UI_1theme.append(self.UI)

        return user_utterance, self.UI



