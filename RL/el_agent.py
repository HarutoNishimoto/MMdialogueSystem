import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import pandas as pd




# ソフトマックス関数
# coefは推定値の振れ幅を調整するためのもの．（デフォルトは1）
def softmax(a, coef=1):
    # 一番大きい値を取得
    c = np.max(a)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(coef * (a - c))
    sum_exp_a = np.sum(exp_a)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a
    return y 


class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.dialogue_log = []
        self.params = defineClass.params()
        self.change_topic_index = 10

    # epsilon以下でランダムな行動，それ以外はQに従った行動
    # softmax=Trueで確率的に選択するようにできます
    def policy(self, s, actions, n_exchg, selection='argmax'):

        # 対話中の1交換目なら話題提示する
        if n_exchg == 0:
            return len(actions)-1
        elif selection == 'argmax':
            if np.random.random() < self.epsilon:
                return np.random.randint(len(actions))
            else:
                if s in self.Q and sum(self.Q[s]) != 0:
                    return np.argmax(self.Q[s])
                else:
                    return np.random.randint(len(actions))
        elif selection == 'softmax':
            if np.random.random() < self.epsilon:
                return np.random.randint(len(actions))
            else:
                if s in self.Q and sum(self.Q[s]) != 0:
                    return np.argmax(softmax(self.Q[s]))
                else:
                    return np.random.randint(len(actions))



    # actionに基づいた発話選択（ランダム選択）
    def utterance_selection(self, action, history_utterance, history_theme):

        # 話題変更した直後は専用の発話を使用する
        if action == self.change_topic_index:
            history_theme.changeTheme()
            next_sysutte = ' *** これから{}の話をしましょう***'.format(history_theme.nowTheme)
            history_utterance.add_sysutte_class('change_theme')
        else:
            # 発話候補の決定
            df = pd.read_csv(self.params.get('path_utterance_by_class'))
            CANDIDATEdf = df[(df['cls'] == action) &
                ((df['theme'] == history_theme.nowTheme) | (df['theme'] == 'default'))]

            # 選択した発話が条件に適するまで変更して決定
            correct = False
            while not correct:
                # もし候補発話がないならクラス関係なく全体から選択
                if len(CANDIDATEdf) == 0:
                    CANDIDATEdf = df[(df['theme'] == history_theme.nowTheme) | (df['theme'] == 'default')]
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]

                # 一旦選択し，適不適を判断
                zantei_df = CANDIDATEdf.sample()
                zantei_index = zantei_df.index[0]
                zantei_sysutte, zantei_theme, zantei_cls = zantei_df.values[0]

                # つかった発話は除外
                if zantei_theme != 'default':
                    if zantei_sysutte in history_utterance.history_sysutte:
                        CANDIDATEdf = CANDIDATEdf.drop(index=[zantei_index])
                    else:
                        next_sysutte, next_sysutte_cls = zantei_sysutte, zantei_cls
                        history_utterance.add_sysutte(next_sysutte, zantei_theme)
                        history_utterance.add_sysutte_class(next_sysutte_cls)
                        correct = True
                else:
                    next_sysutte, next_sysutte_cls = zantei_sysutte, zantei_cls
                    history_utterance.add_sysutte(next_sysutte, 'default')
                    history_utterance.add_sysutte_class(next_sysutte_cls)
                    correct = True
        return next_sysutte


    def init_log(self):
        self.reward_log = []
        self.dialogue_log = []

    def append_log_reward(self, reward):
        self.reward_log.append(reward)

    def append_log_dialogue(self, exchgID, da, theme, impression, s_utte, u_utte):
        self.dialogue_log.append([exchgID+'_S', da, theme, '-', s_utte])
        self.dialogue_log.append([exchgID+'_U', '-', '-', impression, u_utte])

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()

    def write_dialogue_log(self, filename):
        df = pd.DataFrame(data=self.dialogue_log, columns=['exchgID', 'da', 'theme', 'UI', 'utterance'])
        df.to_csv(filename, index=None)

