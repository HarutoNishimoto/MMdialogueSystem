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


# 興味(regression)を入力としてPOMとBASEの行動ラベルを作成
def makeAction(index, pred, true, da, csv=False):
    # make df
    df = pd.DataFrame(data=np.array([index, pred, true, da]).transpose(), columns=['name', 'pred', 'true', 'dialogue_act'])
    # init
    POMDP_state, label = [], []
    # userID
    userID = [x.split("_")[0] for x in df['name'].values]
    userID = np.sort(list(set(userID)))

    for ID in tqdm(userID, ascii=True):
        df_cnt = df[df['name'].str.startswith(ID)]
        pred_cnt, label_cnt, da_cnt = df_cnt['pred'].values, df_cnt['true'].values, df_cnt['dialogue_act'].values
        POMDP_state_cnt = [''] * len(pred_cnt)
        state = np.ones((7, 1))
        interest = main.interest()
        next_state = state
        for i, (p, da) in enumerate(zip(pred_cnt, da_cnt)):
            interest.update(p)
            next_state = cf.updateStatePOMDP(next_state, interest, da)
            exp = np.dot(next_state.flatten(), np.arange(1, len(next_state.flatten())+1, 1))
            POMDP_state_cnt[i] = exp
            label_cnt[i] = label_cnt[i]
        POMDP_state.extend(POMDP_state_cnt)
        label.extend(label_cnt)
    # BASE
    BASE_state = [''] * len(pred)
    for i, val in enumerate(pred):
        BASE_state[i] = val

    # write
    data = np.array([index, POMDP_state, BASE_state, label]).transpose()
    df = pd.DataFrame(data=data, columns=['name', 'POM_state', 'BASE_state', 'label'])
    if type(csv) is str:
        df.to_csv(csv, index=None)
    return df['POM_state'].values, df['BASE_state'].values



# 推定した状態がどのように遷移しているのかをグラフ表示
def drawTrans(pom, base, true, userID):
    xmin, xmax = 1, len(true)

    plt.figure()
    x = range(1, len(true)+1)
    plt.plot(x, pom, label='POMDP', linestyle="dashed")
    plt.plot(x, base, label='BASE', linestyle="dashed")
    plt.plot(x, true, label=r'$L_{UI3}$')
    plt.xlim(xmin, xmax)
    plt.xlabel('exchange index')
    plt.ylabel(r'value of $b(s_1)$')
    plt.title('{}'.format(userID))
    plt.legend()
    plt.rcParams["font.size"] = 30
    plt.show()



# 使えないexchgを外部から指定してcsvから削除
def remNOUSE(df, f_nouse='./refData/noUseExchg.txt'):
    remExchg = pd.read_csv(f_nouse, header=None)
    remExchg = remExchg.iloc[:, 0].values
    for exchg in remExchg:
        df = df[df['name'] != exchg]
    return df

def RMSE(true, pred, printing=True):
    rmse = np.sqrt(np.mean((true-pred)**2))
    if printing:
        print(rmse)
    return rmse



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
    UIdf = pd.read_csv('/Users/haruto/Desktop/mainwork/codes/MLresult/pred_UI3_svr.csv')
    UIdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/UI3/clustering_sysutte/predDAmodel_4da.csv')
    print(len(UIdf))

    # daを結合
    DAdf = pd.read_csv('./../UI3_su_da.csv')
    DAdf = DAdf.rename(columns={'userID': 'name'})
    DAdf = DAdf[['name', 'dialogue_act']]
    df = pd.merge(UIdf, DAdf, on='name')

    index = df['name'].values
    pred = df['fusion_pred'].values
    true = df['label'].values
    da = df['dialogue_act'].values

    # BASEとPOMの比較（最終結果の比較）
    if options.action == 'compare':
        alpha = np.arange(0, 1.1, 0.1)
        sigma = np.arange(0.1, 2.1, 0.1)

        base, pom, param = [], [], []
        for a_num in alpha:
            for s_num in sigma:
                print('###########{}-{}#############'.format(str(a_num), str(s_num)))
                cf.setParam('alpha', a_num)
                cf.setParam('sigma', s_num)
                POM, BASE = makeAction(index, pred, true, da, csv=None)
                base.append(RMSE(true, BASE))
                pom.append(RMSE(true, POM))
                param.append('{}-{}'.format(str(a_num), str(s_num)))

        print('------ best score -------')
        print(param[np.argmin(pom)])



    if options.action == 'compare2':
        # -Iで入力指定しましょう
        df = pd.read_csv(r_file)
        POM = df['POM_state'].values
        BASE2 = df['BASE_state'].values
        true = df['label'].values
        RMSE(BASE2, true)
        print(np.corrcoef(BASE2, true))
        RMSE(POM, true)
        print(np.corrcoef(POM, true))

        ave = np.mean(true)
        BASE1 = [ave] * len(true)
        print('average value : {}'.format(ave))
        RMSE(true, BASE1)
        print(np.corrcoef(true, BASE1))

 
    # 推定したstateの推移をplotする
    if options.action == 'plot':
        print('userID')
        userID = input('>> ')
        df = pd.read_csv(r_file)
        df = df[df['name'].str.startswith(userID)]
        pom = df['POM_state'].values
        base = df['BASE_state'].values
        true = df['label'].values

        print(pom)
        print(base)
        drawTrans(pom, base, true, userID)



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






