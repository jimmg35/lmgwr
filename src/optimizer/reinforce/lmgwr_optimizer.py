import gymnasium as gym
import numpy as np
from typing import Optional, Tuple

from src.model.lmgwr import LMGWR
from src.log.ilogger import ILogger


class LmgwrOptimizerRL(gym.Env):
    """
    PPO environment for optimizing the bandwidth matrix of an LMGWR model.

    Each action adjusts the per-location, per-feature bandwidths. The reward is
    defined as the negative AICc so that minimizing AICc maximizes the reward.
    """

    lmgwr: LMGWR
    logger: ILogger
    min_bandwidth: int
    max_bandwidth: int
    eta: float

    episode_count: int
    reward: float
    remaining_steps: int

    lowest_aicc: float | None
    optimized_r2: float | None
    optimized_bandwidth_matrix: np.ndarray | None

    aicc_records: list[float]
    r2_records: list[float]
    bandwidth_mean_records: list[float]
    bandwidth_variance_records: list[float]

    def __init__(self,
                 lmgwr: LMGWR,
                 logger: ILogger,
                 total_timesteps: int,
                 min_bandwidth: int = 10,
                 max_bandwidth: int = 300,
                 max_steps_per_episode: int = 100,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 eta: float = 0.05):
        super().__init__()
        self.lmgwr = lmgwr
        self.logger = logger
        self.remaining_steps = total_timesteps
        self.eta = eta
        self.lowest_aicc = None
        self.optimized_r2 = None
        self.optimized_bandwidth_matrix = None

        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        n = self.lmgwr.dataset.n
        k = self.lmgwr.dataset.k

        self.action_space = gym.spaces.Box(
            low=min_action,
            high=max_action,
            shape=(n, k),
            dtype=np.int64
        )

        self.observation_space = gym.spaces.Box(
            low=self.min_bandwidth,
            high=self.max_bandwidth,
            shape=(n, k),
            dtype=np.int64
        )

        self.current_bandwidth_matrix = self.__init_bandwidth_matrix(n, k)
        self.__init_step(max_steps_per_episode)

        self.logger.append_info("LmgwrOptimizerRL: environment initialized.")
        self.logger.append_info("LmgwrOptimizerRL: Using AICC as the reward.")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        print(f"- Episode: {self.episode_count} Step {self.current_step}")

        delta = self.__convert_ppo_action_to_bandwidth_adjustment(action)
        self.current_bandwidth_matrix = np.clip(
            self.current_bandwidth_matrix + delta,
            self.min_bandwidth,
            self.max_bandwidth
        ).astype(np.int64)

        self.lmgwr.update_bandwidth_matrix(
            self.current_bandwidth_matrix
        ).exact_fit()

        self.reward = self.__calculate_reward()

        self.current_step += 1
        self.remaining_steps -= 1
        is_max_step_reached = self.current_step >= self.max_steps_per_episode

        if self.lowest_aicc is None:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.lmgwr.r_squared
            self.optimized_bandwidth_matrix = self.current_bandwidth_matrix.copy()

        if abs(self.reward) < self.lowest_aicc:
            self.lowest_aicc = abs(self.reward)
            self.optimized_r2 = self.lmgwr.r_squared
            self.optimized_bandwidth_matrix = self.current_bandwidth_matrix.copy()

        self.aicc_records.append(self.lmgwr.aicc)
        self.r2_records.append(self.lmgwr.r_squared)
        self.bandwidth_mean_records.append(
            float(np.mean(self.current_bandwidth_matrix))
        )
        self.bandwidth_variance_records.append(
            float(np.var(self.current_bandwidth_matrix))
        )

        if is_max_step_reached:
            if self.optimized_r2 is None or self.optimized_bandwidth_matrix is None:
                raise ValueError(
                    "Optimized R2 or bandwidth matrix is None. Check the optimization process."
                )

            bw_summary = (
                f"mean={float(np.mean(self.optimized_bandwidth_matrix)):.4f}, "
                f"var={float(np.var(self.optimized_bandwidth_matrix)):.4f}"
            )
            self.logger.append_bandwidth_optimization(
                self.episode_count,
                self.lowest_aicc,
                self.optimized_r2,
                bw_summary,
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

        return self.current_bandwidth_matrix, self.reward, False, is_max_step_reached, {}

    def reset(self,  # type: ignore[override]
              seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_bandwidth_matrix = self.__init_bandwidth_matrix(
            self.lmgwr.dataset.n,
            self.lmgwr.dataset.k
        )
        self.current_step = 0
        self.episode_count += 1
        self.lowest_aicc = None
        self.optimized_bandwidth_matrix = None

        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []
        print("*** Episode reset ***")
        return self.current_bandwidth_matrix, {}

    def __convert_ppo_action_to_bandwidth_adjustment(self, action: np.ndarray) -> np.ndarray:
        delta = action * (self.max_bandwidth - self.min_bandwidth) * self.eta
        return np.rint(delta).astype(int)

    def __init_bandwidth_matrix(self, n: int, k: int) -> np.ndarray:
        initial_bandwidth = (self.min_bandwidth + self.max_bandwidth) // 2
        matrix = np.full((n, k), initial_bandwidth, dtype=np.int64)
        return matrix

    def __init_step(self, max_steps_per_episode: int) -> None:
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_count = 0

        self.aicc_records = []
        self.r2_records = []
        self.bandwidth_mean_records = []
        self.bandwidth_variance_records = []

    def __calculate_reward(self) -> float:
        return -self.lmgwr.aicc

