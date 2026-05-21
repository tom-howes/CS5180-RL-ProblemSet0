import gymnasium as gym
import numpy as np

class ApartmentEnv(gym.Env):
    def __init__(self, T: int, K: int, seed=None):
        super().__init__()

        self.T = T
        self.K = K
        self.state = None

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([1, 1]), high=np.array([T, K]), dtype=np.int32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t = 1
        u = self.np_random.integers(1, self.K + 1)
        self.state = (t, u)
        return np.array(self.state, dtype=np.int32), {}
    
    def step(self, action: int):
        t, u = self.state

        if action == 1: # Accept
            reward = u
            terminated = True
            next_state = (t, u)
        else:
            reward = 0
            t += 1
            
            if t > self.T: # No weeks left, default to fallback of 0
                next_state = (t, 0)
                reward = 0
                terminated = True
            else: # Draw next apartment
                u_next = self.np_random.integers(1, self.K + 1)
                next_state = (t, u_next)
                terminated = False
        
        self.state = next_state
        truncated = False
        info = {}
        return np.array(next_state, dtype=np.int32), reward, terminated, truncated, info