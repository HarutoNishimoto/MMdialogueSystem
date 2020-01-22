import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import pandas as pd
import pickle
import os
import itertools
from sklearn import preprocessing




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

# ファイルが既に存在する場合，代わりの名前を振ってあげる．
def search_and_rename_filename(oldpath):
    if os.path.exists(oldpath):
        print('file "{}" already exists.'.format(oldpath))
        #dirpath:ディレクトリのパス, filename:対象のファイルまたはディレクトリ
        #name:対象のファイルまたはディレクトリ（拡張子なし）, ext:拡張子
        dirpath, filename = os.path.split(oldpath)
        name, ext = os.path.splitext(filename)

        for i in itertools.count(1):
            newname = '{}_{}{}'.format(name, i, ext)
            newpath = os.path.join(dirpath, newname)
            if not os.path.exists(newpath):
                return newpath
            else:
                print('file "{}" already exists.'.format(newpath))
    else:
        return oldpath

class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.dialogue_log = []
        self.max_n_exchg = 10

    # epsilon以下でランダムな行動，それ以外はQに従った行動
    # softmax=Trueで確率的に選択するようにできます
    def policy(self, s, actions, selection='argmax'):

        if selection == 'argmax':
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
                    return np.argmax(softmax(preprocessing.minmax_scale(self.Q[s])))
                else:
                    return np.random.randint(len(actions))
        else:
            print('invalid "selection"')
            exit(0)


    def init_log(self):
        self.reward_log = []
        self.dialogue_log = []

    def append_log_reward(self, reward):
        self.reward_log.append(reward)

    def append_log_dialogue(self, exchgID, state, action, theme, impression, s_utte, u_utte):
        self.dialogue_log.append([exchgID+'_S', state, action, theme, '-', s_utte])
        self.dialogue_log.append([exchgID+'_U', '-', '-', '-', impression, u_utte])

    def show_reward_log(self, interval=50, episode=-1, filename='sample.png'):
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
            plt.savefig(filename)
            #plt.show()

    def write_dialogue_log(self, filename):
        df = pd.DataFrame(data=self.dialogue_log, columns=['exchgID', 'state', 'action', 'theme', 'UI', 'utterance'])
        filename_new = search_and_rename_filename(filename)
        df.to_csv(filename_new, index=None)
        print('finished making file "{}".'.format(filename_new))

    def saveR(self, filename):
        np.save(filename, np.array(self.reward_log))

    def saveQ(self, table, filename):
        with open(filename, mode='wb') as f:
            pickle.dump(dict(table), f)





