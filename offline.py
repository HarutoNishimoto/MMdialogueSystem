# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import controlFST as cf
import main
from optparse import OptionParser
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


#########################################################################################################
############### csvデータを用いてオフラインで推定，分析をするためのいろいろなコード置き場 #####################
#########################################################################################################


# 使えないexchgを外部から指定してcsvから削除
def remNOUSE(df, f_nouse='./refData/noUseExchg.txt'):
    remExchg = pd.read_csv(f_nouse, header=None)
    remExchg = remExchg.iloc[:, 0].values
    for exchg in remExchg:
        df = df[df['name'] != exchg]
    return df


# 興味(regression)を入力としてPOMとBASEの行動ラベルを作成
# 引数weightは削除していいですよ
def makeAction(index, pred, true, csv=False, thres=None, weight=1):
    # make df
    df = pd.DataFrame(data=np.array([index, pred, true]).transpose(), columns=['name', 'pred', 'true'])
    # init
    POMDP_action, POMDP_state, label = [], [], []
    # chgTopicIdx
    df_idx = pd.read_csv('./refData/topicChgIdx.txt')
    IDX = df_idx.iloc[:, 1:].values
    # userID
    userID = [x.split("_")[0] for x in df['name'].values]
    userID = list(dict.fromkeys(userID))
    # system action
    actionDF = pd.read_csv('190527_sysutte_da.csv')##
    actionDF = actionDF.rename(columns={'userID':'name'})##
    df = pd.merge(df, actionDF, on='name')##

    for ID, idx in tqdm(zip(userID, IDX), ascii=True):
        df_cnt = df[df['name'].str.startswith(ID)]
        pred_cnt = df_cnt['pred'].values
        label_cnt = df_cnt['true'].values
        action_cnt = df_cnt['190527_da'].values##
        
        POMDP_action_cnt = [''] * len(pred_cnt)
        POMDP_state_cnt = [''] * len(pred_cnt)
        state = np.array([[0.5], [0.5]])
        interest = main.interest()
        next_state = state
        for i, (p, action) in enumerate(zip(pred_cnt, action_cnt)):##
            interest.update(p)
            if i+1 in idx:
                next_state = cf.updateStatePOMDPwithAction(weight, next_state, action, interest, True)
            else:
                next_state = cf.updateStatePOMDPwithAction(weight, next_state, action, interest, False)

            POMDP_state_cnt[i] = next_state[0][0]
            next_action = cf.policy(next_state[0][0], thres=thres)
            POMDP_action_cnt[i] = next_action
            label_cnt[i] = label_cnt[i] # LR
        POMDP_action.extend(POMDP_action_cnt)
        POMDP_state.extend(POMDP_state_cnt)
        label.extend(label_cnt)
    # BASE
    BASE_action, BASE_state = [''] * len(pred), [''] * len(pred)
    for i, val in enumerate(pred):
        next_action = cf.policy(val, thres=thres)
        BASE_action[i] = next_action
        BASE_state[i] = val
    # write
    data = np.array([index, POMDP_action, POMDP_state, BASE_action, BASE_state, label]).transpose()
    df = pd.DataFrame(data=data, columns=['name', 'POM', 'POM_state', 'BASE', 'BASE_state', 'label'])
    df = remNOUSE(df)       ### remove no use
    if type(csv) is str:
        df.to_csv(csv, index=None)
    return df['POM_state'].values, df['BASE_state'].values



# 複数のアノテータ結果からラベルに変換
def makeActionLabel(f_action='./refData/topicContinue.csv'):
    # load thres
    thres1, thres2 = cf.getParams('topic_continue')
    #正解のDHC
    df = pd.read_csv(f_action)
    chgTopic = df.iloc[:, 1:].values
    truelabel = [''] * len(df)

    for i, val in enumerate(chgTopic):
        ave = np.mean(chgTopic[i])
        if thres2 <= ave:
            truelabel[i] = 'dig'
        elif (thres1 <= ave) and (ave < thres2):
            truelabel[i] = 'hold'
        else:
            truelabel[i] = 'change'
    return truelabel

# スコアだす
def printScore(true, pred, name='??'):
    coef = cf.getParams('score')
    label = ["dig", "hold", "change"]

    print('############ {} ##############'.format(name))
    print(classification_report(true, pred, labels=label))
    print('-----accuracy-----')
    acc = accuracy_score(true, pred)
    print(acc)
    print('-----CM-----')
    cm = confusion_matrix(true, pred, labels=label)
    print(cm)
    print('-----weighted score-----')
    score = np.sum(coef * cm)
    print(score)

    return acc, score

def RMSE(true, pred, printing=True):
    rmse = np.sqrt(np.mean((true-pred)**2))
    if printing:
        print(rmse)
    return rmse

def drawTrans(pom, base, true, chgIdx, userID):
    xmin, xmax = 1, len(true)
    ymin, ymax = -0.1, 1.1

    plt.figure()
    x = range(1, len(true)+1)
    plt.plot(x, pom, label='POMDP', linestyle="dashed")
    plt.plot(x, base, label='BASE', linestyle="dashed")
    plt.plot(x, true, label=r'$L_{int}$')
    plt.plot([xmin, xmax], [0.5, 0.5], color='k', linestyle="-.")
    for idx in chgIdx:
        plt.plot([idx, idx], [ymin, ymax], color='r', linestyle="-.")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('exchange index')
    plt.ylabel(r'value of $b(s_1)$')
    plt.title('{}'.format(userID))
    plt.legend()
    plt.rcParams["font.size"] = 30
    plt.show()



if __name__ == '__main__':
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action',
                        help='action type', default=None, type='str')
    optparser.add_option('-I', dest='inputfile',
                        help='input file', default=None, type='str')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        action_type = options.action
        r_file = options.inputfile

    #### load
    #### 全部使うかどうかでここ変更してください．
    df = pd.read_csv('pred_inte_unknownLR.csv')
    #df = pd.read_csv('pred_svr_lateDADAfusion.csv')
    index = df['name'].values
    pred = df['fusion_pred'].values
    true = df['label'].values

    ### LRの結果を使いたいなら
    df_LR = pd.read_csv('pred_inte_LR1.txt')
    LR_pred_1 = df_LR['p_prob'].values
    df_LR = pd.read_csv('pred_inte_LR2.txt')
    LR_pred_2 = df_LR['p_prob'].values


    # BASEとPOMの比較（最終結果の比較）
    if options.action == 'compare':
        using_pred_data = LR_pred_1

        #for w in np.arange(1.0, 2.0, 0.1):
        for w in [1.3]:
            print('##########################')
            print(w)
            POM, BASE = makeAction(index, using_pred_data, true, csv='pred_state_11111.csv', weight=w)
            df_cnt = remNOUSE(df)
            true_cnt = df_cnt['label'].values
            RMSE(true_cnt, BASE)
            RMSE(true_cnt, POM)



    if options.action == 'compare2':
        # 入力指定しましょう
        df = pd.read_csv(r_file)
        POM = df['POM_state'].values
        BASE = df['BASE_state'].values
        true = df['label'].values
        RMSE(BASE, true)
        print(np.corrcoef(BASE, true))
        RMSE(POM, true)
        print(np.corrcoef(POM, true))

        


    if options.action == 'compare3':
        # 本当のベースライン（labelの平均をとり続けるもの）
        df = pd.read_csv(r_file)
        true = df['label'].values
        ave = np.mean(true)
        
        baseline = [ave] * len(true)
        RMSE(true, baseline)    # 0.374
        print(np.corrcoef(true, baseline))


        

 
    # 推定したstateの推移をplotする
    if options.action == 'plot':
        print('userID')
        userID = input('>> ')
        df = pd.read_csv('./refData/topicChgIdx.txt')
        idx = df[df['ID'] == userID].values[0, 1:]
        df = pd.read_csv(r_file)
        df = df[df['name'].str.startswith(userID)]
        pom = df['POM_state'].values
        base = df['BASE_state'].values
        true = df['label'].values
        drawTrans(pom, base, true, idx, userID)



    # 一番いいユーザを見つける
    if options.action == 'bestuser':
        df = pd.read_csv(r_file)
        df = remNOUSE(df)

        userID = pd.read_csv('./refData/userID.txt')
        userID = userID.iloc[:, 0].values

        scores = []
        for ID in userID:
            df_cnt = df[df['name'].str.startswith(ID)]
            true = df_cnt['label'].values
            pred_base = df_cnt['BASE_state'].values
            pred_pom = df_cnt['POM_state'].values
            RMSE_base = RMSE(true, pred_base, printing=False)
            RMSE_pom = RMSE(true, pred_pom, printing=False)
            scores.append([ID, RMSE_pom, RMSE_base, RMSE_base - RMSE_pom])

        df = pd.DataFrame(data=np.array(scores), columns=['name', 'pom', 'base', 'diff'])
        df_s = df.sort_values('base')
        df.to_csv('190528_score.csv', index=None)
        print(df_s)

    # 相関をとる
    if options.action == 'sokan':
        def getRegCoef(x, y, sokan):
            std_x = np.std(x)
            std_y = np.std(y)
            a = sokan * (std_y / std_x)
            b = np.mean(y) -(a * np.mean(x))
            return a, b

        df = pd.read_csv('pred_state.csv')

        # select using user
        df['len_name'] = df['name'].apply(lambda x: len(str(x).replace(' ', '')))
        df = df[df['len_name'] == 9]

        pom = df['POM_state'].values
        base = df['BASE_state'].values
        label = df['label'].values

        df = pd.read_csv('./refData/topicContinue.csv')
        action = df.iloc[:, 1:].values
        actionlabel = np.mean(action, axis=1)

        # 横軸の設定（書き換えの必要あり(POM,BASE,label)）
        interest = pom

        x = np.array([interest, actionlabel])
        print('sokan'),
        print(np.corrcoef(x)[0, 1])
        sokan = np.corrcoef(x)[0, 1]

        a, b = getRegCoef(interest, actionlabel, sokan)
        x = np.arange(0, 1.05, 0.05)
        y = a * x + b

        plt.figure()
        plt.scatter(interest, actionlabel)
        plt.plot(x, y, color='r')
        plt.xlim(0, 1)
        plt.ylim(1, 7)
        plt.xlabel('interest level', fontsize=14)
        plt.ylabel('action level', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.show()



    if options.action == 'area':
        df = pd.read_csv('pred_state.csv')
        num = np.arange(0.0, 1.1, 0.1)
        RMSE = []
        xlabel = []
        for i, val in enumerate(num[:-1]):
            df_cnt = df[(num[i] <= df['label']) & (df['label'] < num[i+1])]
            true = df_cnt['label'].values
            pred = df_cnt['POM_state'].values

            rmse = np.sqrt(np.mean((true-pred)**2))
            RMSE.append(rmse)
            print(rmse)
            print(len(df_cnt))

            xlabel.append(str(num[i])[:3] + '-' + str(num[i+1])[:3])

        plt.plot(num[:-1], RMSE)
        plt.ylabel('RMSE (L_int and POMDP)')
        plt.xticks(num[:-1], xlabel, rotation=30)
        plt.xlabel('L_int range')
        plt.show()





        

    ######################################################################
    ############### 毎回使用しないけど使う

    # TCaverageのヒストグラム
    if options.action == 'actionhist':
        df = pd.read_csv('./refData/topicContinue.csv')
        action = df.iloc[:, 1:].values
        action = np.mean(action, axis=1)

        plt.figure()
        plt.hist(action, bins=15)
        plt.xlim(1, 7)
        plt.xlabel('action evaluation (average)', fontsize=18)
        plt.ylabel('frequency', fontsize=18)
        plt.legend(fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()












