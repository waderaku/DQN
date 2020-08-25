from dqn.parameter_set import Parameter_Set
import random
from dqn.dqn_main import dqn_argo
import numpy as np
import copy
import time


# PBTループ回数
PBT_LOOP_NUM = 10

# PBTによるパラメーター変更確率
PBT_UPDATE_RATE = 0.7

''' ハイパーパラメーターリスト '''
# ε_greedyのε初期値
PARAMETER_EPS_INIT = [1, 0.9, 0.8, 0.7, 0.3]

# εの下降割合
PARAMETER_EPS_ANNEAL = [0.9995, 0.999, 0.995, 0.992, 0.99]

# εの最低値
PARAMETER_EPS_MIN = [0.3, 0.2, 0.15, 0.1, 0.01]

# 学習率
PARAMETER_LR = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# 割引率
PARAMETER_GAMMA = [0.999, 0.995, 0.99, 0.97, 0.95]

# バッファーのキャパ
PARAMETER_CAPACITY = [100000, 10000, 5000, 1000, 500]

# Q_taregetの更新頻度
PARAMETER_Q_UPDATE = [100, 10, 7, 5, 2]


def initial_parameter_set():
    '''

    PBTで扱うハイパーパラメーターセットを作成する

    '''

    init_eps_init = random.sample(PARAMETER_EPS_INIT, len(PARAMETER_EPS_INIT))

    inti_eps_anneal = random.sample(
        PARAMETER_EPS_ANNEAL, len(PARAMETER_EPS_ANNEAL))

    init_eps_min = random.sample(PARAMETER_EPS_MIN, len(PARAMETER_EPS_MIN))

    init_lr = random.sample(PARAMETER_LR, len(PARAMETER_LR))

    init_gamma = random.sample(PARAMETER_GAMMA, len(PARAMETER_GAMMA))

    init_cap = random.sample(PARAMETER_CAPACITY, len(PARAMETER_CAPACITY))

    init_Q_update = random.sample(PARAMETER_Q_UPDATE, len(PARAMETER_Q_UPDATE))

    paramerter_set_list = []
    for i in range(len(init_eps_init)):
        param_set = Parameter_Set(
            init_eps_init[i],
            inti_eps_anneal[i],
            init_eps_min[i],
            init_lr[i],
            init_gamma[i],
            init_cap[i],
            init_Q_update[i]
        )
        paramerter_set_list.append(param_set)

    return paramerter_set_list


def update_param(ave_reward_list, param_set_list):
    '''

    パラメーターセットの更新を行う

    '''

    def param_transition(index, target_list):

        # 遷移させるか決定
        rand = np.random.rand()
        if rand < PBT_UPDATE_RATE:
            return
        length = len(target_list)
        rand = np.random.rand()
        # 遷移処理
        # value = list[index]
        if index == 0:
            if rand < 0.5:
                return target_list[1]
        elif index == length - 1:
            if rand < 0.5:
                return target_list[length - 2]
        else:
            if rand < 0.5:
                return target_list[index - 1]
            else:
                return target_list[index + 1]

    # 一番低い評価のパラメーターセットを最大のパラメーターセットにディープコピー
    max_index = np.argmax(ave_reward_list)
    min_index = np.argmin(ave_reward_list)
    param_set_list[min_index] = copy.deepcopy(param_set_list[max_index])
    
    # 一番低い奴の更新
    param_set = param_set_list[min_index]
    # eps_init
    eps_init = param_transition(
        PARAMETER_EPS_INIT.index(param_set.eps_init),
        PARAMETER_EPS_INIT
    )
    # eps_anneal
    eps_anneal = param_transition(
        PARAMETER_EPS_ANNEAL.index(param_set.eps_anneal),
        PARAMETER_EPS_ANNEAL
    )
    # eps_min
    eps_min = param_transition(
        PARAMETER_EPS_MIN.index(param_set.eps_min),
        PARAMETER_EPS_MIN
    )
    # lr
    lr = param_transition(
        PARAMETER_LR.index(param_set.lr),
        PARAMETER_LR
    )
    # gamma
    gamma = param_transition(
        PARAMETER_GAMMA.index(param_set.gamma),
        PARAMETER_GAMMA
    )
    # cap
    cap = param_transition(
        PARAMETER_CAPACITY.index(param_set.cap),
        PARAMETER_CAPACITY
    )
    # q_update
    q_update = param_transition(
        PARAMETER_Q_UPDATE.index(param_set.q_update),
        PARAMETER_Q_UPDATE
    )

    param_set.update(eps_init, eps_anneal, eps_min,
                        lr, gamma, cap, q_update)


    # # 各パラメーターセットの各パラメータを一定確率で隣へ変更する
    # for param_set in param_set_list:

    #     # eps_init
    #     eps_init = param_transition(
    #         PARAMETER_EPS_INIT.index(param_set.eps_init),
    #         PARAMETER_EPS_INIT
    #     )
    #     # eps_anneal
    #     eps_anneal = param_transition(
    #         PARAMETER_EPS_ANNEAL.index(param_set.eps_anneal),
    #         PARAMETER_EPS_ANNEAL
    #     )
    #     # eps_min
    #     eps_min = param_transition(
    #         PARAMETER_EPS_MIN.index(param_set.eps_min),
    #         PARAMETER_EPS_MIN
    #     )
    #     # lr
    #     lr = param_transition(
    #         PARAMETER_LR.index(param_set.lr),
    #         PARAMETER_LR
    #     )
    #     # gamma
    #     gamma = param_transition(
    #         PARAMETER_GAMMA.index(param_set.gamma),
    #         PARAMETER_GAMMA
    #     )
    #     # cap
    #     cap = param_transition(
    #         PARAMETER_CAPACITY.index(param_set.cap),
    #         PARAMETER_CAPACITY
    #     )
    #     # q_update
    #     q_update = param_transition(
    #         PARAMETER_Q_UPDATE.index(param_set.q_update),
    #         PARAMETER_Q_UPDATE
    #     )

    #     param_set.update(eps_init, eps_anneal, eps_min,
    #                      lr, gamma, cap, q_update)
    return param_set_list


if __name__ == '__main__':

    # パラメーターセット取得
    param_set_list = initial_parameter_set()

    max_reward = 100

    # PBT LOOP(10世代？)
    for i in range(PBT_LOOP_NUM):

        print(f"------ {i} 世代 ------")

        ave_reward_list = []

        # dqn実行
        for param_set in param_set_list:
            start = time.time()
            ave_reward, max_reward = dqn_argo(param_set, max_reward)
            time_elapsed = time.time() - start
            print(
                f"{param_set_list.index(param_set)} 回目 param_set: {vars(param_set)} reward: {ave_reward} time elapsed: {time_elapsed}")

            ave_reward_list.append(ave_reward)

    # パラメータ更新処理
    param_set_list = update_param(ave_reward_list, param_set_list)
