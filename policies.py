import numpy as np

class RandomPolicy():
    def __init__(self, T, K):
        self.T = T
        self.K = K

    def act(self, obs):
        return np.random.choice([0, 1], p=[1 - 1/self.T, 1/self.T])

class ThresholdPolicy():
    def __init__(self, u_min):
        self.u_min = u_min
    
    def act(self, obs):
        return 1 if obs[1] >= self.u_min else 0

class OptimalPolicy():
    def __init__(self, T, K):
        self.T = T
        self.K = K
        self.lookup = {1: [0, 0, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1], 4: [1, 1, 1, 1]}

    def act(self, obs):
        t = int(obs[0])
        u = int(np.clip(round(obs[1]), 1, self.K))  # handle noisy floats
        return self.lookup[t][u - 1]