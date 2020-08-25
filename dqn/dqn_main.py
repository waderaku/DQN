import gym  # 倒立振子(cartpole)の実行環境
from gym import wrappers  # gymの画像保存
import numpy as np
import tensorflow as tf
from dqn.model import Network
from dqn.agent import Agent
from dqn.replay_buffer import Replay_Buffer
from dqn.parameter_set import Parameter_Set


# エピソード回数
EPISODE_NUM = 100

# NNのバッチサイズ
BATCH_SIZE = 500

# エピソードスルー回数
INIT_EXPLORATION = 30

# 評価する数（最後何個分で評価するか）
# 最後でいいのか？
# 現状破壊されていることを考えると・・・
REWARD_EVALUATION_SIZE = 20

# モデルを保存するのは、何個の平均が強いときにするか
REWARD_SAVE_EVALUATION_SIZE = 3
SAVE_DIRECTORY = "C:\\Users\\satoshi\\Desktop\\masuda\\model\\"

SAVE_FILE = "cart_pole.hs"
def dqn_argo(param_set: Parameter_Set, max_reward):
    # Agentの生成
    netWork = Network(action_dim=2)
    target_network = Network(action_dim=2)
    agent = Agent(network=netWork,
                  target_network=target_network,
                  eps_start=param_set.eps_init,
                  eps_anneal=param_set.eps_anneal,
                  eps_min=param_set.eps_min,
                  lr=param_set.lr,
                  gamma=param_set.gamma
                  )

    # Envの生成
    env = gym.make('CartPole-v0')

    replay_buffer = Replay_Buffer(param_set.cap)

    save_reward_list = []
    reward_list = []
    for i in range(REWARD_SAVE_EVALUATION_SIZE):
        save_reward_list.append(0)
    for i in range(REWARD_EVALUATION_SIZE):
        reward_list.append(0)

    # データ集め(何回ゲームをやるか)
    for i in range(EPISODE_NUM):

        # Envの初期化情報の取得
        state = env.reset()
        done = False

        # エピソード報酬初期化
        episode_reward = 0

        # 1ゲーム終了させる(Envから終了判定もらう)
        while not done:

            if i > INIT_EXPLORATION:
                # Actionをε-greedyで決める
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()

            # Action引数にEnvからS、r,dの情報を引っ張ってくる
            next_state, reward, done, info = env.step(action)

            # エピソード報酬計算
            episode_reward += reward

            # ReplayBufferにaddする
            replay_buffer.add(
                state,
                action,
                next_state,
                reward,
                done
            )

            # StにSt+1を代入（更新処理）
            state = next_state
        loss = tf.constant(0)

        if i > INIT_EXPLORATION:
            # ニューラルネットワーク学習
            sample = replay_buffer.sample(BATCH_SIZE)
            if sample:
                loss = agent.update(replay_buffer.sample(BATCH_SIZE))

            if i % param_set.q_update == 0:
                agent.network_synchronize()
            
            reward_list[i % REWARD_EVALUATION_SIZE] = episode_reward
            
            save_reward_list[i % REWARD_SAVE_EVALUATION_SIZE] = episode_reward
        
            if  sum(save_reward_list) / len(save_reward_list) >= max_reward:
                print("最高記録更新!!!")
                agent.save(SAVE_DIRECTORY + SAVE_FILE)
                max_reward = sum(save_reward_list) / len(save_reward_list)
    return sum(reward_list) / len(reward_list), max_reward
