import gymnasium as gym
import numpy as np
from enum import Enum
from typing import Tuple, Optional

from src.model.mgwr import MGWR
from src.log.ilogger import ILogger


class MgwrOptimizerRL(gym.Env):
    """
    PPO environment for optimizing MGWR bandwidth sets.

    Each action adjusts the feature-level bandwidths that MGWR uses, with the
    reward defined as the negative AICc (lower is better).
    """

    mgwr: MGWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int
    eta: float

    episode_count: int
    reward: float
    remaining_steps: int

    # Tracking the best result of each episode
    lowest_aicc: float | None
    optimized_r2: float | None
    optimized_bandwidth_set: np.ndarray | None

    # Tracking the process of each episode
    aicc_records: list[float]
    r2_records: list[float]
    bandwidth_mean_records: list[float]
    bandwidth_variance_records: list[float]

    def __init__(self,
                 mgwr: MGWR,
                 logger: ILogger,
                 total_timesteps: int,
                 min_bandwidth: int = 10,
                 max_bandwidth: int = 300,
                 max_steps_per_episode: int = 100,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 eta: float = 0.05):
        super().__init__()
        self.mgwr = mgwr
        self.logger = logger
        self.remaining_steps = total_timesteps
        self.eta = eta
        self.lowest_aicc = None
        self.optimized_r2 = None
        self.optimized_bandwidth_set = None

        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        feature_count = self.mgwr.dataset.k

        self.action_space = gym.spaces.Box(
            low=min_action,
            high=max_action,
            shape=(feature_count,),
            dtype=np.int64
        )

        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth,
            high=self.max_bandwidth,
            shape=(feature_count,),
            dtype=np.int64
        )

        self.current_bandwidth_set = self.__init_bandwidth_set(feature_count)
        self.__init_step(max_steps_per_episode)

        self.logger.append_info(
            "MgwrOptimizerRL: environment initialized."
        )
        self.logger.append_info(
            "MgwrOptimizerRL: Using AICC as the reward."
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        print(f"- Episode: {self.episode_count} Step {self.current_step}")

        delta = self.__convert_ppo_action_to_bandwidth_adjustment(action)

        self.current_bandwidth_set = np.clip(
            self.current_bandwidth_set + delta,
            self.min_bandwidth,
            self.max_bandwidth
        )

        self.mgwr.update_bandwidth_set(
            self.current_bandwidth_set.tolist()
        ).exact_fit()

        self.reward = self.__calculate_reward()

        self.current_step += 1
        self.remaining_steps -= 1
        is_max_step_reached = self.current_step >= self.max_steps_per_episode

        if self.lowest_aicc is None:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.mgwr.r_squared
            self.optimized_bandwidth_set = self.current_bandwidth_set.copy()

        if abs(self.reward) < self.lowest_aicc:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.mgwr.r_squared
            self.optimized_bandwidth_set = self.current_bandwidth_set.copy()

        self.aicc_records.append(self.mgwr.aicc)
        self.r2_records.append(self.mgwr.r_squared)
        self.bandwidth_mean_records.append(
            float(np.mean(self.current_bandwidth_set))
        )
        self.bandwidth_variance_records.append(
            float(np.var(self.current_bandwidth_set))
        )

        if is_max_step_reached:
            if self.optimized_r2 is None or self.optimized_bandwidth_set is None:
                raise ValueError(
                    "Optimized R2 or bandwidth set is None. Please check the optimization process."
                )

            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lowest_aicc,
                self.optimized_r2,
                '[' + ', '.join(map(str, self.optimized_bandwidth_set)) + ']',
                f"Episode {self.episode_count} truncated, "
                f"took {self.current_step} steps, "
                f"reward (lowest AICc): {self.lowest_aicc}, "
                f"r2: {self.optimized_r2}"
            )

            self.logger.append_training_process(
                self.episode_count,
                self.aicc_records,
                self.r2_records,
                bandwidth_mean_records=self.bandwidth_mean_records,
                bandwidth_variance_records=self.bandwidth_variance_records
            )

        return self.current_bandwidth_set, self.reward, False, is_max_step_reached, {}

    def reset(self,  # type: ignore[override]
              seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        feature_count = self.mgwr.dataset.k
        self.current_bandwidth_set = self.__init_bandwidth_set(feature_count)
        self.current_step = 0
        self.episode_count += 1
        self.lowest_aicc = None

        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []
        print("*** Episode reset ***")
        return self.current_bandwidth_set, {}

    def __convert_ppo_action_to_bandwidth_adjustment(self, action: np.ndarray) -> np.ndarray:
        delta = action * (self.max_bandwidth - self.min_bandwidth) * self.eta
        return np.rint(delta).astype(int)

    def __init_bandwidth_set(self, feature_count: int) -> np.ndarray:
        initial_bandwidth = (self.min_bandwidth + self.max_bandwidth) // 2
        return np.full(feature_count, initial_bandwidth, dtype=np.int64)

    def __init_step(self, max_steps_per_episode: int) -> None:
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_count = 0

        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []

    def __calculate_reward(self) -> float:
        return -self.mgwr.aicc
