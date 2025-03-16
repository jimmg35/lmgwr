import gymnasium as gym
import numpy as np
from typing import Tuple, Optional

from src.model.lgwr import LGWR
from src.log.ilogger import ILogger


class LgwrOptimizerRL(gym.Env):

    lgwr: LGWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int
    reward_threshold: float
    episode_count: int

    def __init__(self,
                 lgwr: LGWR,
                 logger: ILogger,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps=100,
                 min_action=-10,
                 max_action=10,
                 reward_threshold=0.75
                 ):
        super(LgwrOptimizerRL, self).__init__()
        self.lgwr = lgwr
        self.logger = logger
        self.reward_threshold = reward_threshold

        # The upper and lower bounds of the estimated bandwidth
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Action space: vectorized bandwidth adjustment
        self.action_space = gym.spaces.Box(
            low=min_action, high=max_action,
            shape=(self.lgwr.dataset.x_matrix.shape[0],), dtype=np.int64
        )

        # Observation space: vectorized bandwidth values
        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth, high=self.max_bandwidth,
            shape=(self.lgwr.dataset.x_matrix.shape[0],), dtype=np.int64
        )

        # Initialize bandwidths and steps
        self.current_bandwidth_vector = self.__init_bandwidth_vector()
        self.__init_step(max_steps)

        self.logger.append_info(
            "LgwrOptimizerRL: LgwrOptimizerRL environment is initialized."
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Ensure every action is an integer
        action = np.round(action).astype(int)

        # Update the bandwidth vector with the action
        self.current_bandwidth_vector = np.clip(
            self.current_bandwidth_vector + action,
            self.min_bandwidth, self.max_bandwidth
        )

        # Apply the updated bandwidth vector to LGWR
        self.lgwr.update_local_bandwidth_vector(
            self.current_bandwidth_vector
        ).fit()

        # Compute reward based on R2
        reward = self.__calculate_reward()

        # Stop episode if reward threshold is met
        done = reward >= self.reward_threshold

        # Maximum step constraint
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        if done:
            print(f"■ Episode {self.episode_count} - R2 {reward}")
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                reward,
                '[' + ', '.join(map(str, self.current_bandwidth_vector)) + ']'
            )

        return self.current_bandwidth_vector, reward, done, truncated, {}

    def reset(self,  # type: ignore
              seed: Optional[int] = None
              ) -> Tuple[np.ndarray, dict]:
        """ Reset the environment to the initial state. """
        super().reset(seed=seed)
        self.current_bandwidth_vector = self.__init_bandwidth_vector()
        self.current_step = 0
        self.episode_count += 1
        return self.current_bandwidth_vector, {}

    def __init_bandwidth_vector(self) -> np.ndarray:
        """ Initialize the local bandwidth vector for LGWR with the same initial value. """
        initial_bandwidth = int(np.random.uniform(
            self.min_bandwidth, self.max_bandwidth))
        return np.full(
            self.lgwr.dataset.x_matrix.shape[0], initial_bandwidth, dtype=np.int64
        )

    def __init_step(self, max_steps):
        """ Initialize step counters. """
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_count = 0

    def __calculate_reward(self) -> float:
        """ Compute R2 as the reward. """
        return self.lgwr.r_squared
