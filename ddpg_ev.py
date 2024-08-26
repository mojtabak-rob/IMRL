import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import gymnasium as gym
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from myenv import NLField

if __name__ == '__main__':
    env=NLField(DoF=2)
    target_actor=keras.models.load_model("target_actor.keras")
    internal_model=keras.models.load_model("internal_model.keras")
    prev_state = env.reset()
    x=[]
    y=[]
    rew=[]
    u0=[]
    u1=[]
    pred = []
    for t in range(500):
        tf_prev_state = keras.ops.expand_dims( # type: ignore
            keras.ops.convert_to_tensor(prev_state), 0 # type: ignore
        )
        action = keras.ops.squeeze(target_actor(tf_prev_state)).numpy() # type: ignore
        tf_action = keras.ops.expand_dims( # type: ignore
            keras.ops.convert_to_tensor(action), 0 # type: ignore
        )
        pred_state = keras.ops.squeeze(internal_model([tf_prev_state,tf_action])).numpy() # type: ignore
        
        x.append(prev_state[0])
        y.append(prev_state[1])
        #u = np.clip(u, env.action_space.low, env.action_space.high) # type: ignore
        s,r,d,inf=env.step(action) # type: ignore
        pred_err = linalg.norm(pred_state-s)
        pred.append(pred_err)
        u0.append(action[0])
        u1.append(action[1])
        rew.append(r)
        prev_state = s

    plt.plot(y)
    plt.plot(x)
    plt.plot(u0)
    plt.plot(u1)
    plt.plot(rew)
    plt.plot(pred)

    plt.show()