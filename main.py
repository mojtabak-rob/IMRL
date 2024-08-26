import numpy as np
import tensorflow as tf
from myenv import NLField
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env=NLField(DoF=2)
    s=env.reset()
    x=[]
    y=[]
    rew=[]
    u0=[]
    u1=[]
    for t in range(1000):
        #env.sets(s)
        x.append(s[0])
        y.append(s[1])
        u=[-0.1*s[1]-5*s[0],-0.1*s[0]-5*s[1]]
        u = np.clip(u, env.action_space.low, env.action_space.high) # type: ignore
        s,r,d,inf=env.step(u) # type: ignore
        u0.append(u[0])
        u1.append(u[1])
        rew.append(r)

    plt.plot(u0)
    plt.show()
