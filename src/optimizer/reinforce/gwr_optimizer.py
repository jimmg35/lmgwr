import gymnasium as gym
import numpy as np
from typing import Tuple

from src.model.gwr import GWR
from src.log.ilogger import ILogger


class GwrOptimizerRL(gym.Env):

    gwr: GWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int

    reward_threshold: float

    episode_count: int
    reward: float

    remaining_steps: int

    lowest_aicc: float | None
    optimized_r2: float | None
    optimized_bandwidth: float | None

    aicc_records: list[float] = []
    r2_records: list[float] = []
    # bandwidth_mean_records: list[float] = []
    # bandwidth_variance_records: list[float] = []

    def __init__(self,
                 gwr: GWR,
                 logger: ILogger,
                 reward_threshold,
                 total_timesteps,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps=100,
                 min_action=-10,
                 max_action=10
                 ):
        super(GwrOptimizerRL, self).__init__()
        self.gwr = gwr
        self.logger = logger
        self.reward_threshold = reward_threshold
        self.reward_threshold = reward_threshold
        self.remaining_steps = total_timesteps
        self.lowest_aicc = None
        self.optimized_r2 = None
        self.optimized_bandwidth = None

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

        self.logger.append_info(
            "GwrOptimizerRL: GwrOptimizerRL environment is initialized."
        )
        self.logger.append_info(
            "GwrOptimizerRL: Using AICC as the reward."
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        print(f"- Episode: {self.episode_count} Step {self.current_step}")
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
        self.reward = self.__calculate_reward()

        # in this case, the episode stops when the R2 is greater than 0.75
        done = self.__if_hit_reward_threshold()

        # the maximum steps of training
        self.current_step += 1
        self.remaining_steps -= 1
        truncated = self.current_step >= self.max_steps

        # assign the initial AICc value to lowest_aicc
        if self.lowest_aicc is None:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.gwr.r_squared
            self.optimized_bandwidth = self.current_bandwidth

        # Update the lowest AICc value
        if abs(self.reward) < self.lowest_aicc:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.gwr.r_squared
            self.optimized_bandwidth = self.current_bandwidth

        # Record the process
        self.aicc_records.append(self.gwr.aicc)
        self.r2_records.append(self.gwr.r_squared)

        if truncated:
            # self.logger.append_info(
            #     f"Episode {self.episode_count} truncated, took {self.current_step} steps, remain {self.remaining_steps} steps, reward: {self.reward}, r2: {self.gwr.r_squared}."
            # )
            if self.optimized_r2 is None or self.optimized_bandwidth is None:
                raise ValueError(
                    "Optimized R2 or bandwidth is None. Please check the optimization process."
                )
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lowest_aicc,
                self.optimized_r2,
                float(self.optimized_bandwidth),
                f"Episode {self.episode_count} truncated, took {self.current_step} steps, reward(lowest AICc): {self.lowest_aicc}, r2: {self.optimized_r2}"
            )
            self.logger.append_training_process(
                self.episode_count,
                self.aicc_records,
                self.r2_records,
                bandwidth_mean_records=None,
                bandwidth_variance_records=None
            )

        # if done:
        #     self.logger.append_bandwidth_optimization(
        #         self.episode_count,
        #         self.gwr.aicc,
        #         self.gwr.r_squared,
        #         int(self.current_bandwidth),
        #         f"★ Episode {self.episode_count} done, took {self.current_step} steps, aicc: {self.gwr.aicc}, r2: {self.gwr.r_squared}"
        #     )

        return np.array([self.current_bandwidth]), self.reward, done, truncated, {}

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
        self.episode_count += 1
        self.lowest_aicc = None
        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []
        return np.array([self.current_bandwidth]), {}

    def __init_bandwidth(self):
        """ 
        Initialize the bandwidth of the GWR model. 
        In this case, we use adaptive bandwidth (int).
        """
        initial_bandwidth = (self.min_bandwidth + self.max_bandwidth) // 2
        return int(initial_bandwidth)

    def __init_step(self, max_steps):
        """ 
        Initialize the step of the GWR model. 
        """
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_count = 0

    def __calculate_reward(self) -> float:
        """ 
        Get the AICc of the GWR model.
        """
        return -self.gwr.aicc

    def __if_hit_reward_threshold(self):
        """ 
        Check if the reward is greater than the threshold.
        """
        if self.reward_threshold is None:
            return False
        return abs(self.reward) <= self.reward_threshold
