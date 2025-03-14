import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
from src.model.gwr import GWR


class GwrOptimizerRL(gym.Env):

    gwr: GWR
    min_bandwidth: int
    max_bandwidth: int

    def __init__(self, gwr: GWR, min_bandwidth=10, max_bandwidth=300):
        super(GwrOptimizerRL, self).__init__()
        self.gwr = gwr

        # The upper and lower bounds of the estimated bandwidth
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Action space: single bandwidth value, the agent is allowed to adjust by -2 to 2.
        self.action_space = gym.spaces.Box(
            low=-2, high=2,
            shape=(1,), dtype=np.int64
        )

        # Observation space: the possible sets of bandwidth values
        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth, high=self.max_bandwidth,
            shape=(1,), dtype=np.int64
        )

        # Initialize bandwidth
        self.current_bandwidth = self.__init_bandwidth()
        print("==========init")
        print(self.current_bandwidth)

        self.prev_r2 = float('inf')
        self.max_steps = 100
        self.current_step = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # action = np.round(action).astype(int)

        # update the bandwidth with an action
        self.current_bandwidth = np.clip(
            self.current_bandwidth +
            action[0], self.min_bandwidth, self.max_bandwidth
        )

        # calculate the matrics of gwr with the updated bandwidth
        self.gwr.update_bandwidth(self.current_bandwidth).fit()

        # reward setting, maximize the R2
        r2 = self.gwr.r_squared
        reward = r2

        # the threshold of stopping the training
        done = False
        self.prev_r2 = r2

        # the maximum steps of training
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return np.array([self.current_bandwidth]), reward, done, truncated, {}

    def reset(self,  # type: ignore
              seed: int | None = None,
              options: Optional[dict] = None
              ) -> Tuple[np.ndarray, dict]:
        """ 
        Reset the environment to the initial state.

        Args:
            seed (int): The seed to reset the environment.
            options (dict): The options to reset the environment.

            Returns:
                Tuple[np.ndarray, dict]: The observation of the environment and the information of the environment.

            Raises:
                ValueError: If the kernel is not set up in the GWR model.
        """
        super().reset(seed=seed)
        self.current_bandwidth = self.__init_bandwidth()
        self.prev_r2 = float('inf')
        self.current_step = 0
        return np.array([self.current_bandwidth]), {}

    def __init_bandwidth(self):
        """ Initialize the bandwidth of the GWR model. """
        return np.random.uniform(
            self.min_bandwidth, self.max_bandwidth
        )
