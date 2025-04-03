from stable_baselines3.common.callbacks import BaseCallback
from src.log.ilogger import ILogger


class EpisodeTracker(BaseCallback):
    """
    A custom callback that logs the episode count and step count.
    """

    customLogger: ILogger

    def __init__(self,
                 logger: ILogger,
                 total_timesteps: int,
                 verbose=1
                 ):
        super(EpisodeTracker, self).__init__(verbose)
        self.customLogger = logger
        self.episode_count = 0
        self.step_count = 0
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:

        self.step_count += 1
        reward = self.locals['rewards'][0]
        truncated = self.locals['infos'][0]['TimeLimit.truncated']

        if truncated:
            self.episode_count += 1
            self.total_timesteps -= self.step_count
            self.customLogger.append_info(
                f"Episode {self.episode_count} truncated, took {self.step_count} steps, remain {self.total_timesteps} steps, reward: {reward}."
            )
            self.step_count = 0

        return True
