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
    eta: float
    initial_bandwidth: int

    episode_count: int
    reward: float
    remaining_steps: int

    # for tracking the best result of each episode
    lowest_aicc: float | None
    optimized_r2: float | None
    optimized_bandwidth_vector: np.ndarray | None
    # for tracking the process of each episode
    aicc_records: list[float] = []
    r2_records: list[float] = []
    bandwidth_mean_records: list[float] = []
    bandwidth_variance_records: list[float] = []

    def __init__(self,
                 lgwr: LGWR,
                 logger: ILogger,
                 total_timesteps,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps_per_episode=100,
                 min_action=-1.0,
                 max_action=1.0,
                 eta=0.001,
                 initial_bandwidth=70
                 ):
        super(LgwrOptimizerRL, self).__init__()
        self.lgwr = lgwr
        self.logger = logger
        self.remaining_steps = total_timesteps
        self.eta = eta
        self.initial_bandwidth = initial_bandwidth
        self.lowest_aicc = None
        self.optimized_r2 = None
        self.optimized_bandwidth_vector = None

        # The upper and lower bounds of the estimated bandwidth
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Action space: vectorized bandwidth adjustment
        self.action_space = gym.spaces.Box(
            low=min_action, high=max_action,
            shape=(self.lgwr.dataset.X.shape[0],), dtype=np.int64
        )

        # Observation space: vectorized bandwidth values
        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth, high=self.max_bandwidth,
            shape=(self.lgwr.dataset.X.shape[0],), dtype=np.int64
        )

        # Initialize bandwidths and steps
        self.current_bandwidth_vector = self.__init_bandwidth_vector()
        self.__init_step(max_steps_per_episode)

        self.logger.append_info(
            "LgwrOptimizerRL: LgwrOptimizerRL environment is initialized."
        )

        self.logger.append_info(
            "LgwrOptimizerRL: Using AICC as the reward."
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        # Ensure every action is an integer
        delta = self.__convert_ppo_action_to_bandwidth_adjustment(action)

        # Update the bandwidth vector with the action
        self.current_bandwidth_vector = np.clip(
            self.current_bandwidth_vector + delta,
            self.min_bandwidth, self.max_bandwidth
        )

        # Apply the updated bandwidth vector to LGWR
        self.lgwr.update_local_bandwidth_vector(
            self.current_bandwidth_vector
        ).fit()

        # Compute reward
        self.reward = self.__calculate_reward()

        # Maximum step constraint
        self.current_step += 1
        self.remaining_steps -= 1
        is_max_step_reached = self.current_step >= self.max_steps_per_episode

        # assign the initial AICc value to lowest_aicc
        if self.lowest_aicc is None:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.lgwr.r_squared
            self.optimized_bandwidth_vector = self.current_bandwidth_vector

        # Update the lowest AICc value
        if abs(self.reward) < self.lowest_aicc:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.lgwr.r_squared
            self.optimized_bandwidth_vector = self.current_bandwidth_vector

        # Record the process
        self.aicc_records.append(self.lgwr.aicc)
        self.r2_records.append(self.lgwr.r_squared)
        self.bandwidth_mean_records.append(
            float(np.mean(self.current_bandwidth_vector))
        )
        self.bandwidth_variance_records.append(
            float(np.var(self.current_bandwidth_vector))
        )

        print(
            f"- Episode: {self.episode_count} Step {self.current_step} AICc: {self.lgwr.aicc} R2: {self.lgwr.r_squared}")
        print(f"  - Action taken: {delta}")
        print(f"  - Current bandwidth vector: {self.current_bandwidth_vector}")

        if is_max_step_reached:
            if self.optimized_r2 is None or self.optimized_bandwidth_vector is None:
                raise ValueError(
                    "Optimized R2 or bandwidth is None. Please check the optimization process."
                )

            # Record the optimized AICc, R2, and bandwidth sets of this episode
            # the bandwidth set could be restored and use to fit the LGWR model again
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lowest_aicc,
                self.optimized_r2,
                '[' + ', '.join(map(str, self.optimized_bandwidth_vector)) + ']',
                f"Episode {self.episode_count} truncated, took {self.current_step} steps, reward(lowest AICc): {self.lowest_aicc}, r2: {self.optimized_r2}"
            )

            # Record the overall details of this episode
            # (this can be used for plotting the trend in this episode)
            self.logger.append_training_process(
                self.episode_count,
                self.aicc_records,
                self.r2_records,
                bandwidth_mean_records=self.bandwidth_mean_records,
                bandwidth_variance_records=self.bandwidth_variance_records
            )

        return self.current_bandwidth_vector, self.reward, False, is_max_step_reached, {}

    def reset(self,  # type: ignore
              seed: Optional[int] = None
              ) -> Tuple[np.ndarray, dict]:
        """ Reset the environment to the initial state. """
        super().reset(seed=seed)
        self.current_bandwidth_vector = self.__init_bandwidth_vector()
        self.current_step = 0
        self.episode_count += 1
        self.lowest_aicc = None
        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []
        print("*** Episode reset ***")
        return self.current_bandwidth_vector, {}

    def __convert_ppo_action_to_bandwidth_adjustment(self, action: np.ndarray) -> np.ndarray:
        """ Convert the PPO action to a bandwidth adjustment value. """
        delta = action * (self.max_bandwidth - self.min_bandwidth) * self.eta
        return np.rint(delta).astype(int)

    def __init_bandwidth_vector(self) -> np.ndarray:
        """ Initialize the local bandwidth vector for LGWR with the same initial value. """
        # initial_bandwidth = (self.min_bandwidth + self.max_bandwidth) // 2
        return np.full(
            self.lgwr.dataset.X.shape[0], self.initial_bandwidth, dtype=np.int64
        )

    def __init_step(self, max_steps_per_episode):
        """ Initialize step counters. """
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_count = 0

    def __calculate_reward(self) -> float:
        """ Use AICc as the reward. """
        return -self.lgwr.aicc
