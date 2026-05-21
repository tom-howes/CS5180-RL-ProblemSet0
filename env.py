import gymnasium as gym
import numpy as np

class ApartmentEnv(gym.Env):
    def __init__(self, T: int, K: int, noisy: bool = False, std: float = 1.0, seed=None):
        super().__init__()

        self.T = T
        self.K = K
        self.noisy = noisy
        self.std = std
        self.state = None

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([1, 1]), high=np.array([T, K]), dtype=np.int32)
    
    def _obs(self, t, u):
        if self.noisy:
            return np.array([t, u + np.random.normal(0, self.std)], dtype=np.float32)
        return np.array([t, u], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t = 1
        u = self.np_random.integers(1, self.K + 1)
        self.state = (t, u)
        return self._obs(t, u), {}
    
    def step(self, action: int):
        t, u = self.state

        if action == 1:  # accept
            reward = u   # true quality determines reward
            terminated = True
            next_state = (t, u)
        else:
            reward = 0
            t += 1

            if t > self.T:
                next_state = (t, 0)
                terminated = True
            else:
                u_next = self.np_random.integers(1, self.K + 1)
                next_state = (t, u_next)
                terminated = False

        self.state = next_state
        return self._obs(*next_state), reward, terminated, False, {}