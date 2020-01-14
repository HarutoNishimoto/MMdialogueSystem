import pandas as pd
import sys
import numpy as np
sys.path.append('/Users/haruto/Desktop/mainwork/codes/MMdialogueSystem')
import defineClass
import itertools
from optparse import OptionParser
from defineClass import historySysUtte
import copy
from el_agent import softmax



# 対話環境
class DialogueEnv(historySysUtte):
    """docstring for DialogueEnv"""
    def __init__(self):
        super().__init__()
        self.params = defineClass.params()

        # action
        self.action_df = pd.read_csv(self.params.get('path_class_name'))
        actions = self.action_df['clsname'].values.tolist()
        self.actions = actions

        # actionにindex付け
        self.actionIndex = {}
        for i, val in enumerate(self.actions):
            self.actionIndex[i] = val

        # 状態は「心象」「直前のシステム対話行為」「対話の位置」の組み合わせ
        self.states_sys_da = ['ct','io','re','qs']
        self.states_impression = ['h','n','l']
        self.states_dialogue_position = ['fh','lh']
        self.states = list(itertools.product(self.states_sys_da, self.states_impression, self.states_dialogue_position))
        self.states = ['_'.join(x) for x in self.states]

        # stateにindex付け
        self.stateIndex = {}
        for i, val in enumerate(self.states):
            self.stateIndex[val] = i

        self.thres_low_UI = 3.5
        self.thres_high_UI = 4.5
        self.persist_UI_exchgs = 3
        self.reward_da_df = pd.read_csv(self.params.get('path_reward_da'), index_col=0)
        self.sys_utterance_log = []
        self.user_utterance_log = []
        self.weight_specific_theme = 0.7

    # 初期化のような感じ
    def reset(self):
        super().__init__()
        self.sys_utterance_log = []
        self.user_utterance_log = []
        return self.stateIndex['io_n_fh']

    # 対話行為を簡単な分類に変換(4種類(ct/io/re/qs))
    def get_simple_da_from_sys_utterance(self, sys_utterance):
        df = pd.read_csv(self.params.get('path_simple_da'))
        da = df[df['agent_utterance'] == sys_utterance]['da_simple'].values
        if '***' in sys_utterance:
            simple_da = 'ct'
        else:
            simple_da = da[0]
        return simple_da

    # 入力：交換番号，出力：前半か後半
    def get_dialogue_position(self, exchg_progress):
        if exchg_progress < 0.5:
            position = 'fh'
        else:
            position = 'lh'
        return position

    # 心象を離散化
    def get_impression_level(self, impression):
        if impression < self.thres_low_UI:
            impression_level = 'l'
        elif self.thres_high_UI <= impression:
            impression_level = 'h'
        else:
            impression_level = 'n'
        return impression_level

    # 次のstateを決定
    def get_next_state(self, exchg_progress, impression, sys_utterance):
        # n_stateは簡単のためにgivenとする
        da_simple = self.get_simple_da_from_sys_utterance(sys_utterance)
        impression_level = self.get_impression_level(impression)
        dialogue_pos = self.get_dialogue_position(exchg_progress)
        n_state = self.stateIndex['{}_{}_{}'.format(da_simple, impression_level, dialogue_pos)]
        return n_state

    # 特定話題の選択に重み（weight）をつける
    def weightSpecificTheme(self, df):
        themes = df['theme'].values
        themes = [1-self.weight_specific_theme if t == 'default' else self.weight_specific_theme for t in themes]
        themes = [x/np.sum(themes) for x in themes]
        df = df.reset_index(drop=True)
        select_index = np.random.choice(df.index.values, p=themes)
        return df.loc[select_index]

    # actionに基づいた発話選択（ランダム選択）
    def utterance_selection(self, action, theme):
        # 話題変更のタイミングでは専用の発話を使用する
        if action == 'change_theme':
            next_sysutte = ' *** これから{}の話をしましょう***'.format(theme)
            self.add_sysutte(next_sysutte, action)
        else:
            # 選択する
            df = pd.read_csv(self.params.get('path_utterance_by_class_named'))
            CANDIDATEdf = df[(df['cls'] == action) &
                ((df['theme'] == theme) | (df['theme'] == 'default'))]
            CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
            CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
            # 使えないものを削除
            for i in range(len(CANDIDATEdf)):
                if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                    CANDIDATEdf = CANDIDATEdf.drop(index=[i])
            # 候補が残っていないなら，action気にせず候補を決定
            if len(CANDIDATEdf) == 0:
                CANDIDATEdf = df[(df['theme'] == theme) | (df['theme'] == 'default')]
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
                # 使えないものを削除
                for i in range(len(CANDIDATEdf)):
                    if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                        CANDIDATEdf = CANDIDATEdf.drop(index=[i])
            # 選択して終了
            SELECTdf = self.weightSpecificTheme(CANDIDATEdf)
            next_sysutte, next_theme, next_action = SELECTdf.values
            self.add_sysutte(next_sysutte, next_action)

        return next_sysutte


    # actionに基づいた発話選択（ランダム選択）
    # heatmapにsoftmaxをちゃんと反映させた
    def utterance_selection_softmax(self, action, prob_actions, theme):
        actions = copy.deepcopy(self.actions)
        prob_actions = copy.deepcopy(prob_actions)

        # 話題変更のタイミングでは専用の発話を使用する
        if action == 'change_theme':
            next_sysutte = ' *** これから{}の話をしましょう***'.format(theme)
            self.add_sysutte(next_sysutte, action)
        else:
            done = False
            while not done:
                # 選択する
                action = np.random.choice(actions, p=softmax(prob_actions))
                df = pd.read_csv(self.params.get('path_utterance_by_class_named'))
                CANDIDATEdf = df[(df['cls'] == action) &
                    ((df['theme'] == theme) | (df['theme'] == 'default'))]
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
                # 使えないものを削除
                for i in range(len(CANDIDATEdf)):
                    if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                        CANDIDATEdf = CANDIDATEdf.drop(index=[i])
                # 候補が残っていない
                if len(CANDIDATEdf) == 0:
                    index = np.where(np.array(actions)==action)[0]
                    actions.pop(index[0])
                    prob_actions.pop(index[0])
                # 候補が残っている（選択して終了）
                else:
                    SELECTdf = self.weightSpecificTheme(CANDIDATEdf)
                    next_sysutte, next_theme, next_action = SELECTdf.values
                    self.add_sysutte(next_sysutte, next_action)
                    done = True

        return next_sysutte




if __name__ == '__main__':


    pass



