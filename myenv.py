from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy import linalg

class NLField(gym.Env):
    def __init__(self,DoF=2):

        self.DoF=DoF
        self.dt = 0.01
        
        #self.target=np.zeros(DoF,dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.DoF,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.DoF,), dtype=np.float32)

    def step(self, u):
        obs = self.state
        dt = self.dt
        u = np.clip(u, self.action_space.low, self.action_space.high) # type: ignore
        
        
        a=0.2
        x=obs[0]
        y=obs[1]

        x_dot=a*y+u[0]
        y_dot=a*x+u[1]
        
        lower_bound = self.observation_space.low # type: ignore
        upper_bound = self.observation_space.high # type: ignore
        new = [x+x_dot*dt,y+y_dot*dt]
        #print(obs)
        #print(new)
        #print(lower_bound)
        #print(lower_bound)
        new = np.clip(new, lower_bound, upper_bound)
        self.state=new
        cost=linalg.norm(obs)+linalg.norm(u)*0.05
        
        done=False
        
        return self._get_obs(), -cost, done, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        new = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high, size=self.DoF) # type: ignore
        self.state = new

        return self._get_obs()

    def sets(self,s):
        self.state=s
    
    def _get_obs(self):
        return self.state
    
