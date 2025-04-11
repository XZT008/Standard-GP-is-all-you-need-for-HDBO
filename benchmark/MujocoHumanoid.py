import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gym

import numpy as np

class MujocoHumanoid:
    def __init__(self):
        self.dims = 1003
        self.steps = int(1003 / 17)
        self.lb = -0.4
        self.ub = 0.4
        self._env = gym.make('HumanoidStandup-v2')

    def __call__(self, x: np.ndarray):
        """
        x in [0, 1] * d
        """
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1

        # normalize x
        x = 0.8 * (x - 0.5)
        x = x.reshape(59, 17)
        total_reward = 0.0

        self._env.reset(seed=2025)
        for x_ in x:
            obs, r, done, _ = self._env.step(x_)
            total_reward += r

            if done:
                break

        return total_reward / self.steps

if __name__ == "__main__":
    b = MujocoHumanoid()
    y_ = []
    x_ = []
    for i in range(5000):
        x = np.random.rand(1003)
        y = b(x)
        y_.append(y)
        x_.append(x)
        print(y)

    y_ = np.array(y_)
    print(f"max: {y_.max()}, min: {y_.min()}, mean: {y_.mean()}")