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
def makeAction(index, pred, true, coef, csv=False, thres=None):
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

    for ID, idx in tqdm(zip(userID, IDX), ascii=True):
        df_cnt = df[df['name'].str.startswith(ID)]
        pred_cnt = df_cnt['pred'].values
        label_cnt = df_cnt['true'].values
        POMDP_action_cnt = [''] * len(pred_cnt)
        POMDP_state_cnt = [''] * len(pred_cnt)
        state = np.array([[0.5], [0.5]])
        interest = main.interest()
        next_state = state
        for i, val in enumerate(pred_cnt):
            interest.update(val)
            if i+1 in idx:
                next_state = cf.updateStatePOMDP(next_state, interest, True, coef)
            else:
                next_state = cf.updateStatePOMDP(next_state, interest, False, coef)
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

def RMSE(true, pred):
    rmse = np.sqrt(np.mean((true-pred)**2))
    print(rmse)

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
    plt.savefig('./graph/{}.png'.format(userID))
    print(userID)



if __name__ == '__main__':
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action',
                        help='action type', default=None, type='str')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        action_type = options.action

    #### load
    #### 全部使うかどうかでここ変更してください．
    df = pd.read_csv('pred_LR.csv')
    index = df['name'].values
    pred = df['fusion_pred'].values
    true = df['label'].values

    ### ここ3行はまとめてコメントアウト
    df_LR = pd.read_csv('MM_result_lr-0.005_var-0.025_lam-0.0.txt')
    pred = df_LR['p_prob'].values
    df['fusion_pred'] = pred







    # BASEとPOMの比較（最終結果の比較）
    if options.action == 'compare':

        coef = np.arange(1.9,2.5,0.1)
        coef = [2.0]
        for c in coef:
            POM, BASE = makeAction(index, pred, true, c, csv='pred_state_new.csv')
            # compare
            df = remNOUSE(df)
            print(c)
            RMSE(df['label'].values, POM)   #0.316
            RMSE(df['label'].values, BASE)  #0.319
                                      
    # 推定したstateの推移をplotする
    # ユーザ全部のグラフを保存するようにしました
    if options.action == 'plot':
        userID = pd.read_csv('./refData/userID.txt')
        userID = userID.iloc[:, 0].values
        userID = ['F02']

        for ID in userID:
            df = pd.read_csv('./refData/topicChgIdx.txt')
            idx = df[df['ID'] == ID].values[0, 1:]

            df = pd.read_csv('pred_state_new.csv')
            df = df[df['name'].str.startswith(ID)]
            pom = df['POM_state'].values[:]
            base = df['BASE_state'].values[:]
            true = df['label'].values[:]
            drawTrans(pom, base, true, idx, ID)



    # 一番いいユーザを見つける
    if options.action == 'bestuser':
        df = pd.read_csv('pred_state.csv')
        df = remNOUSE(df)

        userID = pd.read_csv('./refData/userID.txt')
        userID = userID.iloc[:, 0].values
        userID = [x.split('_')[0] for x in userID]
        userID = list(set(userID))

        scores = []
        for ID in userID:
            df_cnt = df[df['name'].str.startswith(ID)]

            pred_pom = df_cnt['POM_state'].values
            pred_base = df_cnt['BASE_state'].values
            true = df_cnt['label'].values
            RMSE_pom = np.sqrt(np.mean((true-pred_pom)**2))
            RMSE_base = np.sqrt(np.mean((true-pred_base)**2))
            scores.append([ID, RMSE_pom, RMSE_base, RMSE_base - RMSE_pom])

        df = pd.DataFrame(data=np.array(scores), columns=['name', 'pom', 'base', 'diff'])
        df_s = df.sort_values('diff')
        print(df_s.iloc[:, [0, 3]])

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



    if options.action == 'rmse':
        df = pd.read_csv('pred_LR.csv')
        true = df['label'].values

        df = pd.read_csv('MM_lr-0.05_var-0.025_lam-0.0_itr-200.txt')
        meta = input('>> ')
        meta = 'p_prob'
        pred = df[meta].values

        rmse = np.sqrt(np.mean((true-pred)**2))
        print(rmse)



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


    if options.action == 'area2':

        unko = main.interest(3)
        unko.update(300)
        unko.update(300)
        print(unko.get_prev_ave_interest())

        



            





    ######################################################################
    ############### 毎回使用しないけど使う

    # 正解の行動のヒストグラム
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





















    #################################################################################
    ####### ここしようしないです． ############################################


    # 政策関数のグリッドサーチ
    if options.action == 'grid_policy':
        para1 = np.arange(0.2, 0.45, 0.05)
        para2 = np.arange(0.6, 0.85, 0.05)
        pom_acc, base_acc, pom_score, base_score, para = [], [], [], [], []
        meta = ['PARAM','POM_ACC','BASE_ACC', 'POM_SCORE', 'BASE_SCORE']

        #正解の行動のread
        truelabel = makeActionLabel()  
        for p1 in para1:
            for p2 in para2:
                #predする
                POM, BASE = makeAction(index, pred, true, csv='pred_state.csv', thres=[p1,p2])
                # score
                pom_acc.append(printScore(truelabel, POM, 'POM')[0])
                pom_score.append(printScore(truelabel, POM, 'POM')[1])
                base_acc.append(printScore(truelabel, BASE, 'BASE')[0])
                base_score.append(printScore(truelabel, BASE, 'BASE')[1])
                para.append(str(p1) + '_' + str(p2))

        data = np.array([para, pom_acc, base_acc, pom_score, base_score]).transpose()
        df = pd.DataFrame(data=data, columns=meta)
        df.to_csv('grid.csv', index=None)


    # 正解の行動ごとの推定した状態のヒストグラム
    if options.action == 'POMhist':
        truelabel = makeActionLabel()

        df = pd.read_csv('action_pred.csv')
        df['action'] = truelabel

        df_dig = df[df['action'] == 'dig']
        dig = df_dig['POM_state'].values
        df_hold = df[df['action'] == 'hold']
        hold = df_hold['POM_state'].values
        df_change = df[df['action'] == 'change']
        change = df_change['POM_state'].values

        plt.figure()
        plt.hist(dig, label='dig', alpha=0.4)
        plt.hist(hold, label='hold', alpha=0.4)
        plt.hist(change, label='change', alpha=0.4)
        plt.legend()
        plt.show()

    # （グリッドサーチ後）正解率で2手法比較
    if options.action == 'vsACC':
        df = pd.read_csv('grid.csv')
        pomacc = df['POM_ACC'].values
        baseacc = df['BASE_ACC'].values
        pomscore = df['POM_SCORE'].values
        basescore = df['BASE_SCORE'].values
        x = np.arange(1, len(pomacc)+1, 1)

        plt.figure()
        plt.plot(x, pomacc, label='pom')
        plt.plot(x, baseacc, label='base')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(x, pomscore, label='pom')
        plt.plot(x, basescore, label='base')
        plt.legend()
        plt.show()

