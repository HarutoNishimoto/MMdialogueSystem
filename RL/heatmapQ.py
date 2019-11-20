import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing


##### これは自作もの＃＃＃＃＃

def show_q_value(Q, state_name, action_name):
    # 例外処理
    action_name[-1] = 'CT'

    state_size = len(state_name)      # row
    action_size = len(action_name)    # col
    reward_map = np.zeros((state_size, action_size))

    # 辞書を2次元リストに変換
    for s in range(state_size):   #state_size
        for a in range(action_size):   #action_size
            if s in Q.keys():
                reward_map[s][a] = Q[s][a]
    # 正規化
    reward_map_normed = preprocessing.minmax_scale(reward_map, axis=1)

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.7, hspace=0.7)

    for i, (r_map, title) in enumerate(zip([reward_map, reward_map_normed], ['reward_map', 'reward_map_normed'])):
        ax = fig.add_subplot(1, 2, i+1)
        plt.imshow(r_map, cmap=cm.RdYlGn, interpolation="bilinear",
                   vmax=abs(r_map).max(), vmin=-abs(r_map).max())
        # 表示する値の範囲
        ax.set_xlim(-0.5, action_size - 0.5)
        ax.set_ylim(-0.5, state_size - 0.5)
        # 表示するメモリの値
        ax.set_xticks(np.arange(action_size))
        ax.set_yticks(np.arange(state_size))
        # メモリにラベルを貼る
        ax.set_xticklabels(action_name, rotation=90, fontsize=8)
        ax.set_yticklabels(state_name, fontsize=8)
        # 軸のラベル
        ax.set_xlabel('action')
        ax.set_ylabel('state')
        ax.set_title(title)
        ax.grid(which="both")
    plt.show()
