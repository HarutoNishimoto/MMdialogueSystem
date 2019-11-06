# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import controlBelief as cb
import defineClass
from optparse import OptionParser
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# 基本的に，python predOffline.py -A offlineで動く．
# モデルを指定するときは，optionでつけれるはず

# システム発話を入力として，(class, theme)を出力する
def getUtteranceClassTheme(utterance):

    params = defineClass.params()
    classFile = params.get('path_utterance_by_class')
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


if __name__ == '__main__':
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action',
                        help='action type', default=None, type='str')
    optparser.add_option('--model', dest='model',
                        help='euclid_dim3? euclid_dim1? base? hand_STP?', default='euclid_dim3_weighted', type='str')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        model = options.model

    # パラメータ管理
    params = defineClass.params()

    ### オフライン．1発話ずつユーザ発話とUIを入力していくもの
    if options.action == 'offline':
        # init
        UI = defineClass.userImpression()
        utteHis = defineClass.historySysUtte()
        themeHis = defineClass.historyTheme(random_choice=False)
        UWordHis = defineClass.historyUserWord()
        init_UI3 = 4
        current_belief, next_belief = np.ones((7, 1)), np.ones((7, 1))
        exchg_num = 15

        df = pd.DataFrame(data=[], columns=['exchgID', 'speaker', 'class', 'theme', 'UI3', 'utterance'])
        try:
            for i in range(exchg_num):
                # POMDPでb(s)を更新
                current_belief = next_belief
                if i == 0:
                    UI.update(init_UI3)
                    next_belief = cb.updateBelief(current_belief, UI, 'change_topic', params)
                    themeHis.decideNextTheme(init_UI3)
                    next_sysUtte, action = cb.Policy(next_belief, None, UWordHis, utteHis, themeHis, model=model)
                else:
                    UI.update(current_UI3)
                    next_belief = cb.updateBelief(current_belief, UI, action, params)
                    themeHis.decideNextTheme(current_UI3)
                    next_sysUtte, action = cb.Policy(next_belief, user_utterance, UWordHis, utteHis, themeHis, model=model)
                print(next_sysUtte)
                # ユーザの発話情報を入力
                user_utterance = input('what do you say? >> ')
                current_UI3 = float(input('How is UI3? >> '))
                # log管理
                c, t = getUtteranceClassTheme(next_sysUtte)
                df.loc[str(i).zfill(3)+'_S'] = [str(i).zfill(3), 'S', action, t, '-', next_sysUtte]
                df.loc[str(i).zfill(3)+'_U'] = [str(i).zfill(3), 'U', '-', '-', current_UI3, user_utterance]
            df.to_csv('samplelog.csv', index=None)

        except KeyboardInterrupt:
            print('\n終了します．')
            df.to_csv('samplelog.csv', index=None)



    # 作成したlog見てannotation
    if options.action == 'annotation':

        logfilename = 'samplelog.csv'
        basename, filetype = logfilename.split('.')
        df = pd.read_csv(logfilename)

        exchg_num = int(len(df)/2)
        annotation = []
        for ID in range(exchg_num):

            print('交換番号{}です．'.format(str(ID).zfill(3)))
            print(df.loc[ID*2, 'utterance'])
            print(df.loc[ID*2+1, 'utterance'])
            anno = input('annotation >> ')
            annotation.append(anno)

        print('名前を入力してください．')
        name = input('>> ')

        ANNOdf = pd.DataFrame(
            data=np.array([list(set(df['exchgID'].values)), annotation]).transpose(),
            columns=['exchgID', 'annotation'])
        ANNOdf.to_csv(basename+'_'+name+'.'+filetype, index=None)










