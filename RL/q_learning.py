import sys
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import pandas as pd
import numpy as np
import copy
import pickle
import os
import MeCab
import el_agent
import virtual_user
from defineClass import historySysUtte
from el_agent import ELAgent
from heatmapQ import show_q_value, make_risouQ
from heatmapQ import Qsum
from optparse import OptionParser
from collections import defaultdict
from el_agent import softmax
from dialogue_env import DialogueEnv


### https://github.com/openai/gym/blob/master/gym/core.py
### ここ見てクラスとか作成しましょう


## 入力のファイル
## １．クラス情報がかかれたファイル
## ２．そのクラスを簡易的な対話行為にすると何に該当するかを書いたファイル


# システム側
class QlearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    # rewardを定義
    def get_reward(self, env, UserModel, state, next_state, action_name, n_exchg):
        # state(id)からstate_name(str)をreturn
        def get_state_name(d, state, index):
            values = [k for k, v in d.items() if v == state]
            if index == None:
                return values[0]
            else:
                return values[0].split('_')[index]

        # 現在の心象に応じて決定
        if get_state_name(env.stateIndex, state, index=2) == 'h':
            r_UI = options.R_oneUI
        elif get_state_name(env.stateIndex, state, index=2) == 'n':
            r_UI = 0
        elif get_state_name(env.stateIndex, state, index=2) == 'l':
            r_UI = -options.R_oneUI

        # N連続で心象を見る
        r_PUI = 0
        if len(UserModel.log_UI_1theme) >= env.persist_UI_exchgs:
            persist_UI = np.array(UserModel.log_UI_1theme[-env.persist_UI_exchgs:])
            if np.count_nonzero(persist_UI >= env.thres_high_UI) == env.persist_UI_exchgs:
                r_PUI = options.R_persistUI
            elif np.count_nonzero(persist_UI <= env.thres_low_UI) == env.persist_UI_exchgs:
                r_PUI = -options.R_persistUI

        # 対話行為のbigramで報酬
        da1 = get_state_name(env.stateIndex, state, index=0)
        da2 = get_state_name(env.stateIndex, next_state, index=0)
        if da2 == 'ct':
            r_DA = 0
        else:
            r_DA = env.reward_da_df.loc[da1, da2]

        # 固有名詞に適切に反応できたら正の報酬
        properNoun_response = ['qs_o_d','qs_o_s','re_o_m']
        noun_presence = get_state_name(env.stateIndex, state, index=1)
        if action_name in properNoun_response:
            if noun_presence == 'No':
                r_NOUN = options.R_noun
            elif noun_presence == 'Nx':
                r_NOUN = -options.R_noun
        else:
            r_NOUN = 0

        # 感謝に対しては取り締まる
        if (action_name == 'thank') and (action_name != 'change_theme'):
            r_TNK = options.R_thank
        else:
            r_TNK = 0

        # 報酬の総和
        reward = r_UI+ r_PUI+ options.Rc_bigram * r_DA + r_NOUN + r_TNK
        return reward

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=10, coef_epsilon=0.99):

        self.init_log()
        actions = list(env.actionIndex.keys())
        self.Q = defaultdict(lambda: [0] * len(actions))
        self.Q_n_update = defaultdict(lambda: [0] * len(actions))

        ep = self.epsilon
        for e in range(episode_count):
            self.epsilon = max(coef_epsilon ** e, ep)

            s = env.reset()#reset
            themeHis = defineClass.historyTheme(random_choice=True)#reset
            UserModel = virtual_user.UserModel()#reset

            rewards = []
            for n_exchg in range(self.max_n_exchg):
                if n_exchg == 0:
                    chg_theme, theme = themeHis.decideNextTheme(None)
                else:
                    chg_theme, theme = themeHis.decideNextTheme(impression)
                # システム発話決定
                if chg_theme:
                    a_name = 'change_theme'
                else:
                    a = self.policy(s, actions, selection='argmax')# 発話クラス選択
                    a_name = env.actionIndex[a]
                sys_utterance = env.utterance_selection(a_name, theme)# 発話選択
                env.sys_utterance_log.append(sys_utterance)
                user_utterance, impression = UserModel.getResponse(sys_utterance)# 応答選択
                env.user_utterance_log.append(user_utterance)
                n_state = env.get_next_state(impression, sys_utterance, user_utterance)# 次のstateを決める
                reward = self.get_reward(env, UserModel, s, n_state, a_name, n_exchg)# 報酬計算

                if not chg_theme:# Q更新
                    gain = reward + gamma * max(self.Q[n_state])
                    estimated = self.Q[s][a]
                    self.Q[s][a] += learning_rate * (gain - estimated)
                    self.Q_n_update[s][a] += 1
                s = n_state
                rewards.append(reward)
            else:
                self.append_log_reward(np.mean(rewards))

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)




# システム側（学習済み）
class TrainedQlearningAgent(ELAgent):
    def __init__(self, filename):
        super().__init__(epsilon=0)

        # 学習すみQテーブルの読み込み
        with open(filename, mode='rb') as f:
            self.Q = pickle.load(f)

    # Qの学習されていないところを埋める
    def fillQ(self, env):
        for k in range(len(env.states)):
            if k not in self.Q.keys():
                self.Q[k] = [0] * len(env.actions)
            else:
                pass

    # システム発話を入力として，(class, theme)を出力する
    def getUtteranceClassTheme(self, utterance):
        params = defineClass.params()
        classFile = params.get('path_utterance_by_class_named')
        themeFile = params.get('path_theme_info')
        CLSdf = pd.read_csv(classFile)
        THEMEdf = pd.read_csv(themeFile)

        if '***' in utterance:
            return '-', '-'
        else:
            clsInfo = CLSdf[CLSdf['agent_utterance'] == utterance]['cls'].values.astype('str')
            clsInfo = '-'.join(clsInfo)
            themeInfo = THEMEdf[THEMEdf['agent_utterance'] == utterance]['theme'].values[0]
            return clsInfo, themeInfo

    def conversation(self, env):
        self.init_log()
        actions = list(env.actionIndex.keys())

        s = env.reset()#reset
        themeHis = defineClass.historyTheme(random_choice=False)#reset
        for n_exchg in range(self.max_n_exchg):
            if n_exchg == 0:
                chg_theme, theme = themeHis.decideNextTheme(None)
            else:
                chg_theme, theme = themeHis.decideNextTheme(impression)

            sys_utterance = env.utterance_selection_softmax(chg_theme, theme, self.Q[s], coef=5)# 発話選択
            env.sys_utterance_log.append(sys_utterance)
            print(sys_utterance)
            user_utterance = input('your utterance >> ')# 発話入力
            impression = float(input('your impression >> '))# 心象入力
            env.user_utterance_log.append(user_utterance)
            n_state = env.get_next_state(impression, sys_utterance, user_utterance)

            states = [k for k, v in env.stateIndex.items() if v == s]
            self.append_log_dialogue(str(n_exchg).zfill(2),
                states[0],
                env.history_sysutte_class[-1],
                self.getUtteranceClassTheme(sys_utterance)[1],
                impression,
                sys_utterance,
                user_utterance)

            # 更新
            s = n_state


if __name__ == "__main__":
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action', default=None, type='str')
    optparser.add_option('--model', dest='model', default='sample', type='str')
    optparser.add_option('--ep', dest='n_episode', default=1000, type='int')
    optparser.add_option('--seed', dest='seed', default=777, type='int')
    optparser.add_option('--alpha', dest='alpha', default=0.1, type='float')
    optparser.add_option('--interval', dest='interval', default=10, type='int')
    optparser.add_option('--epsilon', dest='epsilon', default=0.1, type='float')
    optparser.add_option('--coef_epsilon', dest='coef_epsilon', default=0.99, type='float')

    optparser.add_option('--R_oneUI', dest='R_oneUI', default=1, type='int')
    optparser.add_option('--R_persistUI', dest='R_persistUI', default=5, type='int')
    optparser.add_option('--Rc_bigram', dest='Rc_bigram', default=0.5, type='float')
    optparser.add_option('--R_noun', dest='R_noun', default=10, type='int')
    optparser.add_option('--R_thank', dest='R_thank', default=-10, type='int')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        print('############\n{}\n############'.format(str(options)))

    # seed
    np.random.seed(options.seed)

    ######## python q_learning.py -A [ACT] --model [MODEL] ##############
    Qtable_name = '{}/{}_Q'.format(options.model, options.model)
    Qfreq_name = '{}/{}_Qfreq'.format(options.model, options.model)
    hm_name = '{}/{}_hm.png'.format(options.model, options.model)
    reward_name = '{}/{}_reward.png'.format(options.model, options.model)
    reward_list_name = '{}/{}_reward.npy'.format(options.model, options.model)
    log_name = '{}/{}_log.csv'.format(options.model, options.model)

    # params
    params = defineClass.params()

    # Qを学習
    if options.action == 'train':
        # dir作成
        if not os.path.exists(options.model):
            os.mkdir(options.model)
        else:
            print('model "{}" already exists.'.format(options.model))
            if_del = input('### overwrite if you push enter. ###')

        env = DialogueEnv()
        agent = QlearningAgent(epsilon=options.epsilon)
        agent.learn(env,
            episode_count=options.n_episode,
            learning_rate=options.alpha,
            coef_epsilon=options.coef_epsilon)
        agent.saveQ(agent.Q, Qtable_name)
        agent.show_reward_log(interval=options.interval, filename=reward_name)
        agent.saveR(reward_list_name)
        show_q_value(agent.Q, env.states, env.actions, hm_name)
        # アクセスした回数を保存
        agent.saveQ(agent.Q_n_update, Qfreq_name)


    # 学習済みQを用いて対話
    if options.action == 'dialogue':
        env = DialogueEnv()
        agent = TrainedQlearningAgent(Qtable_name)
        agent.fillQ(env)
        agent.conversation(env)
        agent.write_dialogue_log(log_name)

    # 理想のQを作成
    if options.action == 'make_risouQ':
        modelname = params.get('path_risouQ')
        # dir作成
        if not os.path.exists(modelname):
            os.mkdir(modelname)
        else:
            print('model "{}" already exists.'.format(modelname))
            if_del = input('### overwrite if you push enter. ###')

        env = DialogueEnv()
        Q_name = '{}/{}'.format(modelname, modelname.split('/')[-1])
        make_risouQ(Q_name, env.states, env.actions)

    # 理想のQを用いて対話
    if options.action == 'dialogue_Q':
        modelname = params.get('path_risouQ')
        Q_name = '{}/{}'.format(modelname, modelname.split('/')[-1])
        log_name = '{}/{}_log.csv'.format(modelname, modelname.split('/')[-1])
        env = DialogueEnv()
        agent = TrainedQlearningAgent(Q_name)
        agent.fillQ(env)
        agent.conversation(env)
        agent.write_dialogue_log(log_name)



    ###### ここから下は今は使用していないです #########
    if options.action == 'chkoption':
        print(options)

