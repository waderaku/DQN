import tensorflow as tf


class Network(tf.keras.models.Model):
    def __init__(self, action_dim, fcs=[256, 256]):
        super().__init__()
        self.layer_set = [tf.keras.layers.Dense(
            hidden, activation="relu") for hidden in fcs]
        self.final_layer = tf.keras.layers.Dense(action_dim)

    def call(self, x):
        for layer in self.layer_set:
            x = layer(x)
        x = self.final_layer(x)
        return x
