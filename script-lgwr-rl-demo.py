import gym
import numpy as np
# from gym import spaces
from stable_baselines3 import PPO
from typing import Tuple, Optional
import gymnasium as gym
from typing import Any


class LgwrEnv(gym.Env):
    def __init__(self, gwr_model, n_points, min_bw=5, max_bw=200):
        super(LgwrEnv, self).__init__()
        self.gwr_model = gwr_model
        self.n_points = n_points
        self.min_bw = min_bw
        self.max_bw = max_bw

        # 動作空間：每個回歸點的帶寬調整
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.n_points,),
            dtype=np.float32
        )

        # 狀態空間：每個回歸點的帶寬
        self.observation_space = gym.spaces.Box(
            low=self.min_bw, high=self.max_bw,
            shape=(self.n_points,),
            dtype=np.float32
        )

        self.current_bw = np.random.uniform(
            self.min_bw, self.max_bw, self.n_points
        )  # 初始化帶寬

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 更新每個回歸點的帶寬
        self.current_bw = np.clip(
            self.current_bw + action, self.min_bw, self.max_bw
        )

        # 計算GWR指標
        aic_values = self.gwr_model.calculate_aic(self.current_bw)
        total_aic = np.sum(aic_values)  # LGWR 的總 AIC

        # 設定獎勵
        reward = -total_aic

        # 結束條件：可設定 AIC 變動小於某閾值
        done = False

        return self.current_bw, reward, done, False, {}

    def reset(self,  # pyright: ignore
              seed: int | None = None,
              options: dict[str, Any] | None = None
              ):
        super().reset(seed=seed)
        self.current_bw = np.random.uniform(
            self.min_bw, self.max_bw, self.n_points
        )
        return self.current_bw, {}


# # 初始化 LGWR 環境
n_points = 100  # 假設有 100 個回歸點
env = LgwrEnv(gwr_model, n_points)

# # 訓練 RL 代理 (使用 PPO)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=50000)  # 需要較長時間訓練

# # 測試最優帶寬
# obs, _ = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs)
#     obs, reward, done, _, _ = env.step(action)
#     print(f"帶寬: {obs}, 獎勵 (AIC): {-reward}")
