import gymnasium as gym
import numpy as np
from enum import Enum
from typing import Tuple, Optional

from src.model.lgwr import LGWR
from src.log.ilogger import ILogger


class LgwrRewardType(Enum):
    R2 = "r2"
    AICC = "aicc"


class LgwrOptimizerRL(gym.Env):

    lgwr: LGWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int

    reward_type: LgwrRewardType
    reward_threshold: float | None

    episode_count: int
    reward: float

    def __init__(self,
                 lgwr: LGWR,
                 logger: ILogger,
                 reward_threshold,
                 reward_type=LgwrRewardType.R2,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps=100,
                 min_action=-10,
                 max_action=10
                 ):
        super(LgwrOptimizerRL, self).__init__()
        self.lgwr = lgwr
        self.logger = logger
        self.reward_type = reward_type
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

        if self.reward_type == LgwrRewardType.R2:
            self.logger.append_info(
                "LgwrOptimizerRL: Using R2 as the reward."
            )
        elif self.reward_type == LgwrRewardType.AICC:
            self.logger.append_info(
                "LgwrOptimizerRL: Using AICC as the reward."
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

        # Compute reward
        self.reward = self.__calculate_reward()

        # Stop episode if reward threshold is met
        done = self.__if_hit_reward_threshold()

        # Maximum step constraint
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        if done:
            print(
                f"★ Episode {self.episode_count} done, took {self.current_step} steps, {self.reward_type}: {self.lgwr.aicc}"
            )
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lgwr.aicc,
                self.lgwr.r_squared,
                '[' + ', '.join(map(str, self.current_bandwidth_vector)) + ']'
            )

        return self.current_bandwidth_vector, self.reward, done, truncated, {}

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
        if self.reward_type == LgwrRewardType.R2:
            return self.lgwr.r_squared
        elif self.reward_type == LgwrRewardType.AICC:
            return -self.lgwr.aicc
        else:
            raise ValueError("Invalid reward type.")

    def __if_hit_reward_threshold(self):
        """ Define the reward type. """
        if self.reward_threshold is None:
            return False
        if self.reward_type == LgwrRewardType.R2:
            return self.reward >= self.reward_threshold
        elif self.reward_type == LgwrRewardType.AICC:
            return abs(self.reward) <= self.reward_threshold
        else:
            raise ValueError("Invalid reward type.")
