import numpy as np

import tensorflow as tf


class Agent:
    def __init__(
        self,
        network,
        target_network,
        eps_start=1,
        eps_anneal=0.9995,
        eps_min=0.15,
        lr=1e-3,
        gamma=0.99,
    ):
        self.gamma = tf.constant(gamma)
        self.network = network
        self.target_network = target_network
        self.eps = eps_start
        self.eps_anneal = eps_anneal
        self.eps_min = eps_min
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def get_action(self, state):
        rand = np.random.rand()
        self.eps = np.max([self.eps*self.eps_anneal, self.eps_min])
        if self.eps > rand:
            return np.random.randint(0, 2)
        else:
            shape = (1,) + state.shape
            return np.argmax(self.network(state.reshape(shape)))

    def save(self, path):
        self.network.save(path)

    @tf.function
    def update(self, sample_data):
        [state, action, next_state, reward, done] = sample_data

        # calculate target = gamma * Q(s_(t+1), a_(t+1)) + r_(t+1)
        # with shape(batch_size, action_dim)
        Q_target_values = self.target_network(next_state)
        Q_values = self.network(next_state)

        # with shape(batch_size, 1)
        # Q_next = tf.math.reduce_max(Q_target_values, axis=1, keepdims=True)
        tf_range = tf.reshape(tf.range(action.shape[0]), (-1, 1))
        Q_max_action = tf.math.argmax(
            Q_values, axis=1, output_type=tf.int32)
        next_indice = tf.concat(
            [tf_range, tf.reshape(Q_max_action, (-1, 1))], axis=1)

        Q_target_values = tf.gather_nd(self.network(state), next_indice)
        target = (1 - done) * self.gamma * Q_target_values + reward

        # [[0, a_0], [1, a_1], ..., [batch_size-1, a_last]]
        indice = tf.concat([tf_range, action], axis=1)

        # start auto gradient
        with tf.GradientTape() as tape:

            Q = tf.gather_nd(self.network(state), indice)
            # calculate loss
            loss = tf.math.reduce_mean(tf.keras.losses.MSE(target, Q))
            gradients = tape.gradient(loss, self.network.trainable_variables)
            # apply_gradients
            self.optimizer.apply_gradients(
                zip(gradients, self.network.trainable_variables))
        # for evaluation
        return loss

    def network_synchronize(self):
        self.target_network.set_weights(self.network.get_weights())
