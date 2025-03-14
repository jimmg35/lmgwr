import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
from src.model.gwr import GWR


class GwrOptimizerRL(gym.Env):

    gwr: GWR
    min_bandwidth: int
    max_bandwidth: int

    def __init__(self,
                 gwr: GWR,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps=100,
                 min_action=-10,
                 max_action=10):
        super(GwrOptimizerRL, self).__init__()
        self.gwr = gwr

        # The upper and lower bounds of the estimated bandwidth
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Action space: single bandwidth value, the agent is allowed to adjust by -2 to 2.
        self.action_space = gym.spaces.Box(
            low=min_action, high=max_action,
            shape=(1,), dtype=np.int64
        )

        # Observation space: the possible sets of bandwidth values
        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth, high=self.max_bandwidth,
            shape=(1,), dtype=np.int64
        )

        # Initialize bandwidth, steps of the agent
        self.current_bandwidth = self.__init_bandwidth()
        self.__init_step(max_steps)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # ensure every action is an integer
        # (comply with the adaptive bandwidth nature)
        action = np.round(action).astype(int)

        # update the bandwidth with an action
        self.current_bandwidth = np.clip(
            self.current_bandwidth + action[0],
            self.min_bandwidth, self.max_bandwidth
        )

        # calculate the matrics of gwr with the updated bandwidth
        self.gwr.update_bandwidth(self.current_bandwidth).fit()

        # reward setting, maximize the R2
        reward = self.__calculate_reward()

        # the threshold of stopping the training
        # (False means non-stop)
        # in this case, the episode stops when the R2 is greater than 0.75
        done = reward >= 0.75

        # the maximum steps of training
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return np.array([self.current_bandwidth]), reward, done, truncated, {}

    def reset(self,  # type: ignore
              seed: int | None = None,
              ) -> Tuple[np.ndarray, dict]:
        """ 
        Reset the environment to the initial state.

        Args:
            seed (int): The seed to reset the environment.

            Returns:
                Tuple[np.ndarray, dict]: The observation of the environment and the information of the environment.

            Raises:
                ValueError: If the kernel is not set up in the GWR model.
        """
        super().reset(seed=seed)
        self.current_bandwidth = self.__init_bandwidth()
        self.current_step = 0
        return np.array([self.current_bandwidth]), {}

    def __init_bandwidth(self):
        """ 
        Initialize the bandwidth of the GWR model. 
        In this case, we use adaptive bandwidth (int).
        """
        return int(np.random.uniform(
            self.min_bandwidth, self.max_bandwidth
        ))

    def __init_step(self, max_steps):
        """ 
        Initialize the step of the GWR model. 
        """
        self.max_steps = max_steps
        self.current_step = 0

    def __calculate_reward(self) -> float:
        """ 
        Get the R2 of the GWR model.
        """
        return self.gwr.r_squared
