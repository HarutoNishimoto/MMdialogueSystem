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
    def get_reward(self, env, UserModel, state, next_state, action_name):
        # 文から固有名詞を抽出
        def getProperNoun(sentence):
            mt = MeCab.Tagger()
            node = mt.parseToNode(sentence)
            properNouns = []
            while node:
                fields = node.feature.split(",")
                # 名詞、動詞、形容詞に限定
                if (fields[0] == '名詞') and (fields[1] == '固有名詞'):
                    properNouns.append(node.surface)
                node = node.next
            return properNouns
        
        # state(index)からstate_name(str)をreturn
        def get_state_name(d, state, index=1):
            #心象についての状態は{0}_{1}_{2}でいう1に書かれている
            values = [k for k, v in d.items() if v == state]
            if index == None:
                return values[0]
            else:
                return values[0].split('_')[index]

        # 現在の心象に応じて決定
        if get_state_name(env.stateIndex, state, index=1) == 'h':
            r_UI = options.R_oneUI
        elif get_state_name(env.stateIndex, state, index=1) == 'n':
            r_UI = 0
        elif get_state_name(env.stateIndex, state, index=1) == 'l':
            r_UI = -options.R_oneUI

        # N連続で心象を見る
        r_PUI = 0
        if len(UserModel.log_UI_1theme) >= env.persist_UI_exchgs:
            persist_UI = np.array(UserModel.log_UI_1theme[-env.persist_UI_exchgs:])
            if np.count_nonzero(persist_UI > env.thres_high_UI) == env.persist_UI_exchgs:
                r_PUI = options.R_persistUI
            elif np.count_nonzero(persist_UI < env.thres_low_UI) == env.persist_UI_exchgs:
                r_PUI = -options.R_persistUI

        # 対話行為のbigramで報酬
        da1 = get_state_name(env.stateIndex, state, index=0)
        da2 = get_state_name(env.stateIndex, next_state, index=0)
        if da2 == 'ct':
            r_DA = 0
        else:
            r_DA = env.reward_da_df.loc[da1, da2]

        '''
        # 固有名詞に適切に反応できたらボーナス
        properNoun_response = ['qs_cont','re_nocont']
        properNouns = getProperNoun(UserModel.log_user_utterance_1theme[-1])
        if (len(properNouns) > 0) and (action_name in properNoun_response):
            r_PN = 100
        else:
            r_PN = 0
        '''

        # 報酬の総和
        reward = r_UI+ r_PUI+ options.Rc_bigram * r_DA
        return reward

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=10, max_n_exchg=10):
        print('######\nepisode_count : {}\nepsilon : {}\nlearning_rate : {}\n######'
            .format(str(episode_count), str(self.epsilon), str(learning_rate)))

        self.init_log()
        actions = list(env.actionIndex.keys())
        self.Q = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()#reset
            themeHis = defineClass.historyTheme()#reset
            UserModel = virtual_user.UserModel()#reset
            done = False
            for n_exchg in range(max_n_exchg):
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
                n_state = env.get_next_state(float(n_exchg)/max_n_exchg, impression, sys_utterance)# 次のstateを決める
                reward = self.get_reward(env, UserModel, s, n_state, a_name)# 報酬計算

                if not chg_theme:# Q更新
                    gain = reward + gamma * max(self.Q[n_state])
                    estimated = self.Q[s][a]
                    self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
            else:
                self.append_log_reward(reward)

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

    def conversation(self, env, max_n_exchg=10):
        self.init_log()
        actions = list(env.actionIndex.keys())

        s = env.reset()#reset
        themeHis = defineClass.historyTheme()#reset
        for n_exchg in range(max_n_exchg):
            if n_exchg == 0:
                chg_theme, theme = themeHis.decideNextTheme(None)
            else:
                chg_theme, theme = themeHis.decideNextTheme(impression)

            if chg_theme:
                a_name = 'change_theme'
            else:
                a = self.policy(s, actions, selection='argmax')# 発話クラス選択
                a_name = env.actionIndex[a]
            sys_utterance = env.utterance_selection_softmax(a_name, self.Q[s], theme)# 発話選択
            env.sys_utterance_log.append(sys_utterance)
            print(sys_utterance)
            user_utterance = input('what do you say? >> ')# 発話入力
            impression = float(input('your UI3? >> '))# 心象入力
            env.user_utterance_log.append(user_utterance)
            n_state = env.get_next_state(n_exchg, impression, sys_utterance)

            # 更新
            s = n_state

            self.append_log_dialogue(str(n_exchg).zfill(3),
                env.history_sysutte_class[-1],
                self.getUtteranceClassTheme(sys_utterance)[1],
                impression,
                sys_utterance,
                user_utterance)


if __name__ == "__main__":
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action', default=None, type='str')
    optparser.add_option('--model', dest='model', default='sample', type='str')
    optparser.add_option('--ep', dest='n_episode', default=1000, type='int')
    optparser.add_option('--seed', dest='seed', default=777, type='int')
    optparser.add_option('--alpha', dest='alpha', default=0.1, type='float')
    optparser.add_option('--interval', dest='interval', default=10, type='int')

    optparser.add_option('--R_oneUI', dest='R_oneUI', default=10, type='int')
    optparser.add_option('--R_persistUI', dest='R_persistUI', default=50, type='int')
    optparser.add_option('--Rc_bigram', dest='Rc_bigram', default=5.0, type='float')
    #optparser.add_option('--R_doubleuse', dest='R_doubleuse', default=-30, type='int')
    #optparser.add_option('--R_context', dest='R_context', default=100, type='int')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        print('interval : {}'.format(options.interval))
        print('seed : {}'.format(options.seed))
    # seed
    np.random.seed(options.seed)

    ######## python q_learning.py -A [ACT] --model [MODEL] ##############
    Qtable_name = '{}/{}_Q'.format(options.model, options.model)
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
        agent = QlearningAgent(epsilon=0.1)
        agent.learn(env,
            episode_count=options.n_episode,
            learning_rate=options.alpha)
        agent.saveQ(Qtable_name)
        agent.show_reward_log(interval=options.interval, filename=reward_name)
        agent.saveR(reward_list_name)
        show_q_value(agent.Q, env.states, env.actions, hm_name)

    # 学習済みQを用いて対話
    if options.action == 'dialogue':
        env = DialogueEnv()
        agent = TrainedQlearningAgent(Qtable_name)
        agent.fillQ(env)
        agent.conversation(env, max_n_exchg=10)
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
        agent.conversation(env, max_n_exchg=1)
        agent.write_dialogue_log(log_name)




    ###### ここから下は今は使用していないです #########


    if options.action == 'chkoption':
        print(options)

    if options.action == 'multi-train':
        multitime = 5
        Qarray = []
        for i in range(multitime):
            ## 複数回する時の条件設定
            modelname = './{}/{}of{}'.format(options.model, str(i+1), multitime)
            np.random.seed(i+1)
            Qtable_name = modelname + '_Q'
            hm_name = modelname + '_hm.png'
            reward_name = modelname + '_reward.png'
            reward_list_name = modelname + '_reward.npy'
            log_name = modelname + '_log.csv'

            env = DialogueEnv()
            agent = QlearningAgent(epsilon=0.1)
            agent.learn(env,
                episode_count=options.n_episode,
                learning_rate=options.alpha)
            show_q_value(agent.Q, env.states, env.actions, hm_name)
            Qarray.append(agent.Q)

        hm_name = './{}/{}_hm.png'.format(options.model, options.model)
        Qtable_name = './{}/{}_Q'.format(options.model, options.model)
        Qave = Qsum(Qarray, len(env.states), len(env.actions))
        agent.Q = Qave
        agent.saveQ(Qtable_name)
        show_q_value(agent.Q, env.states, env.actions, hm_name)




