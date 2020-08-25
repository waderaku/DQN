import gym
import tensorflow as tf
import numpy as np
import time

SAVE_DIRECTORY = "C:\\Users\\satoshi\\Desktop\\masuda\\model\\"
SAVE_FILE = "cart_pole.hs"

env = gym.make("CartPole-v0")

state = env.reset()
done = False
index = 0
network = tf.keras.models.load_model(SAVE_DIRECTORY + SAVE_FILE)
while not done:
    index += 1
    shape = (1,) + state.shape
    action = np.argmax(network(state.reshape(shape)))
    next_state, reward, done, info = env.step(action)
    env.step(action)
    env.render()
    time.sleep(0.01)
    state = next_state
    if done:
        print(index)