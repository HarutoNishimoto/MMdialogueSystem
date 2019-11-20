from collections import defaultdict
import gym
from el_agent import ELAgent
import pandas as pd
import sys
import numpy as np
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import virtual_user
import pickle
from heatmapQ import show_q_value
import itertools
import el_agent


### https://github.com/openai/gym/blob/master/gym/core.py
### ここ見てクラスとか作成しましょう


# 特定の文字列(state)を含むkeyのvalueをreturn
def get_value_from_part_state(d, state):
    # state => 'str'
    values = [v for k, v in d.items() if state in k]
    return values

# 対話行為を簡単な3種類に分類
def get_simple_da_from_sys_utterance(sys_utterance):
    df = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem/da5.csv')
    da = df[df['agent_utterance'] == sys_utterance]['da5_new'].values

    # simple_da => ['ct', io','re','qu']
    if '***' in sys_utterance:
        simple_da = 'ct'
    else:
        if da[0] == 'information_offer':
            simple_da = 'io'
        elif 'response' in da[0]:
            simple_da = 're'
        elif 'question' in da[0]:
            simple_da = 'qs'
        else:
            simple_da = 'io'

    return simple_da




# 対話環境
class DialogueEnv():
    """docstring for DialogueEnv"""
    def __init__(self, modelname):

        # action読み込み（str）
        df = pd.read_csv('191024_DAname.txt', header=None)
        actions = df[0].values.tolist() + ['change_theme']
        self.actions = actions

        # actionにindex付け
        self.actionIndex = {}
        for i, val in enumerate(self.actions):
            self.actionIndex[i] = val

        # 状態は心象（h/n/l）と直前のシステム対話行為（i/r/q）の組み合わせ
        self.states_impression = ['high','normal','low']
        self.states_sys_da = ['ct','io','re','qs']
        self.states = list(itertools.product(self.states_impression, self.states_sys_da))
        self.states = ['_'.join(x) for x in self.states]

        # stateにindex付け
        self.stateIndex = {}
        for i, val in enumerate(self.states):
            self.stateIndex[val] = i

        self.n_exchg_1step = 10
        self.thres_low_UI = 3.5
        self.thres_high_UI = 4.5
        self.thres_low_UI_average = 3.5

    # 初期化のような感じ
    def reset(self):
        return self.stateIndex['normal_io']


    # 対話を1つ前に進める
    def step(self, action, state, n_exchg, UserModel, sys_utterance):
        # 1ステップは一定交換したらdoneとする
        if n_exchg >= self.n_exchg_1step:
            done = True
        else:
            done = False

        ##### rewardを定義
        # state(とaction)に応じて決定
        if state in get_value_from_part_state(self.stateIndex, 'high'):
            r_UI = 1
        elif state in get_value_from_part_state(self.stateIndex, 'normal'):
            r_UI = 0.5
        elif state in get_value_from_part_state(self.stateIndex, 'low'):
            r_UI = -0.5

        # UIが低い時に話題変更したら報酬が高い
        if action == len(self.actions)-1:
            if np.average(UserModel.log_UI_1theme) < self.thres_low_UI_average:
                r_TC = 2.0
            else:
                r_TC = 0.0
        else:
            r_TC = 0.0
        ##### rewardを定義

        # n_state(next_state)はPOMで推定
        # 今は簡単のためにstateはgivenとする
        user_utterance, impression = UserModel.getResponse(sys_utterance)
        da_simple = get_simple_da_from_sys_utterance(sys_utterance)
        #print(user_utterance)
        if impression < self.thres_low_UI:
            n_state_impression = 'low'
        elif self.thres_high_UI <= impression:
            n_state_impression = 'high'
        else:
            n_state_impression = 'normal'
        n_state = self.stateIndex[n_state_impression + '_' + da_simple]

        # 報酬の総和
        reward = r_UI + r_TC
        return n_state, reward, done

    # 対話を1つ前に進める
    # 学習済みのモデルを動かす
    def step_trained(self, n_exchg, sys_utterance, impression):
        # 1ステップは一定交換したらdoneとする
        if n_exchg >= self.n_exchg_1step:
            done = True
        else:
            done = False

        # UIをもとに現在のstateを決定
        da_simple = get_simple_da_from_sys_utterance(sys_utterance)
        if impression < self.thres_low_UI:
            n_state_impression = 'low'
        elif self.thres_high_UI <= impression:
            n_state_impression = 'high'
        else:
            n_state_impression = 'normal'
        n_state = self.stateIndex[n_state_impression + '_' + da_simple]

        return n_state, done







# システム側
class DialogueAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=10):
        self.init_log()
        actions = list(env.actionIndex.keys())
        self.Q = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()#reset
            utteHis = defineClass.historySysUtte()#reset
            themeHis = defineClass.historyTheme_for_RL()#reset
            UserModel = virtual_user.UserModel()#reset
            done = False
            n_exchg = 0
            while not done:
                # システム発話決定
                a = self.policy(s, actions, n_exchg, selection='softmax')# 発話クラス選択
                sys_utterance = self.utterance_selection(a, utteHis, themeHis)# 発話選択
                #print(sys_utterance)
                # ユーザ発話決定
                n_state, reward, done = env.step(a, s, n_exchg, UserModel, sys_utterance)# 報酬を決定，次のstateを予測（ここを練る）
                # テーブルの更新
                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                n_exchg += 1

            else:
                self.append_log_reward(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

    # Qテーブルを保存
    def saveQ(self):
        print('Qを保存します．\n保存するファイル名を入力してください．')
        filename = input('>> ')
        with open(filename, mode='wb') as f:
            pickle.dump(dict(agent.Q), f) 


# システム側（学習済み）
class TrainedDialogueAgent(ELAgent):

    def __init__(self, filename):
        super().__init__(epsilon=0)

        # 学習すみQテーブルの読み込み
        with open(filename, mode='rb') as f:
            self.Q = pickle.load(f)


    def conversation(self, env, max_n_exchg=10):
        self.init_log()
        actions = list(env.actionIndex.keys())

        s = env.reset()#reset
        utteHis = defineClass.historySysUtte()#reset
        themeHis = defineClass.historyTheme_for_RL()#reset
        n_exchg = 0
        for i in range(max_n_exchg):
            # システム発話決定
            a = self.policy(s, actions, i, selection='softmax')# 発話クラス選択
            sys_utterance = self.utterance_selection(a, utteHis, themeHis)# 発話選択
            print(sys_utterance)
            # ユーザ発話入力
            user_utterance = input('what do you say? >> ')
            current_impression = float(input('How is UI3? >> '))
            n_state, done = env.step_trained(n_exchg, sys_utterance, current_impression)
            # 更新
            s = n_state
            n_exchg += 1

            self.append_log_dialogue(str(i).zfill(3),
                utteHis.get_prev_sysutte_class(),
                themeHis.nowTheme,
                current_impression,
                sys_utterance,
                user_utterance)




if __name__ == "__main__":

    Qtable_name = '191119_Qtable_softmax_coef-1.pickle'
    model_name = '191024_fea15_norm'
    log_name = 'sample.csv'

    # 自作のものを動かす（train）
    if sys.argv[1] == 'train':
        env = DialogueEnv(model_name)
        agent = DialogueAgent(epsilon=0.1)
        agent.learn(env, episode_count=1000)
        agent.saveQ()
        agent.show_reward_log(interval=10)

    # 学習済みのモデルを用いて対話
    if sys.argv[1] == 'dialogue':
        env = DialogueEnv(model_name)
        agent = TrainedDialogueAgent(Qtable_name)
        agent.conversation(env, max_n_exchg=10)
        agent.write_dialogue_log(log_name)

    # 学習済みのQのヒートマップ見れるよ
    if sys.argv[1] == 'heatmap':
        agent = TrainedDialogueAgent(Qtable_name)
        env = DialogueEnv(model_name)
        show_q_value(agent.Q, env.states, env.actions)


    if sys.argv[1] == 'unti':




        # 学習すみQテーブルの読み込み
        with open(Qtable_name, mode='rb') as f:
            Q = pickle.load(f)

        for k, v in Q.items():
            print(k)
            print(v)
            print(el_agent.softmax(v,coef=1))







