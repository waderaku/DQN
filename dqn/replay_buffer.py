import numpy as np
from collections import namedtuple

import tensorflow as tf

import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class Replay_Buffer:
    def __init__(self, capacity):
        self.buffer_list = []
        self.position = 0
        self.capacity = capacity

    def add(self, state, action, next_state, reward, done):
        if len(self.buffer_list) < self.capacity:
            self.buffer_list.append(None)
        self.buffer_list[self.position] = Transition(
            state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        if len(self.buffer_list) < batch_size:
            return

        # batch_sizezだけランダムに抽出
        sampling_data = random.sample(
            self.buffer_list, batch_size)

        # 返却データの初期化
        state_array = np.zeros((1,) + sampling_data[0].state.shape)
        action_array = np.zeros((1, 1))
        next_array = np.zeros((1,) + sampling_data[0].next_state.shape)
        reward_array = np.zeros((1, 1))
        done_array = np.zeros((1, 1))

        for data in sampling_data:
            state_array = np.vstack([state_array,
                                     data.state])
            action_array = np.vstack([action_array,
                                      data.action])
            next_array = np.vstack([next_array,
                                    data.next_state])
            reward_array = np.vstack([reward_array,
                                      data.reward])
            done_array = np.vstack([done_array,
                                    data.done])

        # tensorflow変換
        state_tf = tf.constant(state_array[1:], dtype=tf.float32)
        action_tf = tf.constant(action_array[1:], dtype=tf.int32)
        next_tf = tf.constant(next_array[1:], dtype=tf.float32)
        reward_tf = tf.constant(reward_array[1:], dtype=tf.float32)
        done_tf = tf.constant(done_array[1:], dtype=tf.float32)

        return [state_tf, action_tf, next_tf, reward_tf, done_tf]
