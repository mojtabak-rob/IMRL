import numpy as np
import tensorflow as tf
from myenv import NLField
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env=NLField(DoF=2)
    s=[5,2]
    x=[]
    y=[]
    rew=[]
    for t in range(1000):
        env.sets(s)
        x.append(s[0])
        y.append(s[1])
        s,r,d,inf=env.step(u=[-0.005*s[1]-0.01*s[0],-0.005*s[0]-0.02*s[1]]) # type: ignore
        rew.append(r)

    plt.plot(rew)
    plt.show()
