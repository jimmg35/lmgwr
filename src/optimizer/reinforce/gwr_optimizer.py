import gymnasium as gym
import numpy as np
from typing import Tuple

from src.model.gwr import GWR
from src.log.ilogger import ILogger


class GwrOptimizerRL(gym.Env):

    model: GWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int
    eta: float

    episode_count: int
    reward: float
    remaining_steps: int

    # for tracking the best result of each episode
    lowest_aicc: float | None
    optimized_r2: float | None
    optimized_bandwidth: float | None
    # for tracking the process of each episode
    aicc_records: list[float] = []
    r2_records: list[float] = []

    def __init__(self,
                 model: GWR,
                 logger: ILogger,
                 total_timesteps,
                 min_bandwidth=10,
                 max_bandwidth=300,
                 max_steps_per_episode=100,
                 min_action=-1.0,
                 max_action=1.0,
                 eta = 0.05
                 ):
        super(GwrOptimizerRL, self).__init__()
        self.model = model
        self.logger = logger
        self.remaining_steps = total_timesteps
        self.eta = eta
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
        self.__init_step(max_steps_per_episode)

        self.logger.append_info(
            "GwrOptimizerRL: GwrOptimizerRL environment is initialized."
        )
        self.logger.append_info(
            "GwrOptimizerRL: Using AICC as the reward."
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        print(f"- Episode: {self.episode_count} Step {self.current_step}")
        
        # ensure every action is an integer
        delta = self.__convert_ppo_action_to_bandwidth_adjustment(action)

        # update the bandwidth with an action
        self.current_bandwidth = np.clip(
            self.current_bandwidth + delta[0],
            self.min_bandwidth, self.max_bandwidth
        )

        # calculate the matrics of gwr with the updated bandwidth
        self.model.update_bandwidth(self.current_bandwidth).fit()

        # reward setting, minimize the AICc
        self.reward = self.__calculate_reward()

        # the maximum steps of training
        self.current_step += 1
        self.remaining_steps -= 1
        is_max_step_reached = self.current_step >= self.max_steps_per_episode

        # assign the initial AICc value to lowest_aicc
        if self.lowest_aicc is None:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.model.r_squared
            self.optimized_bandwidth = self.current_bandwidth

        # Update the lowest AICc value
        if abs(self.reward) < self.lowest_aicc:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.model.r_squared
            self.optimized_bandwidth = self.current_bandwidth

        # Record the process
        self.aicc_records.append(self.model.aicc)
        self.r2_records.append(self.model.r_squared)

        if is_max_step_reached:
            if self.optimized_r2 is None or self.optimized_bandwidth is None:
                raise ValueError(
                    "Optimized R2 or bandwidth is None. Please check the optimization process."
                )
            
            # Record the optimized AICc, R2, and bandwidth sets of this episode
            # the bandwidth set could be restored and use to fit the LGWR model again
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lowest_aicc,
                self.optimized_r2,
                float(self.optimized_bandwidth),
                f"Episode {self.episode_count} truncated, took {self.current_step} steps, reward(lowest AICc): {self.lowest_aicc}, r2: {self.optimized_r2}"
            )
            
            # Record the overall details of this episode 
            # (this can be used for plotting the trend in this episode)
            self.logger.append_training_process(
                self.episode_count,
                self.aicc_records,
                self.r2_records,
                bandwidth_mean_records=None,
                bandwidth_variance_records=None
            )

        return np.array([self.current_bandwidth]), self.reward, False, is_max_step_reached, {}

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

    def __convert_ppo_action_to_bandwidth_adjustment(self, action: np.ndarray) -> np.ndarray:
        """ Convert the PPO action to a bandwidth adjustment value. """
        delta = action * (self.max_bandwidth - self.min_bandwidth) * self.eta
        return np.rint(delta).astype(int)
    
    def __init_bandwidth(self):
        """ 
        Initialize the bandwidth of the GWR model. 
        In this case, we use adaptive bandwidth (int).
        """
        initial_bandwidth = (self.min_bandwidth + self.max_bandwidth) // 2
        return int(initial_bandwidth)

    def __init_step(self, max_steps_per_episode):
        """ 
        Initialize the step of the GWR model. 
        """
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_count = 0

    def __calculate_reward(self) -> float:
        """ 
        Get the AICc of the GWR model.
        """
        return -self.model.aicc

