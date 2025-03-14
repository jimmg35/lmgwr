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
        if self.locals['dones'][0]:
            self.episode_count += 1
            remaining_steps = self.total_timesteps - self.num_timesteps
            self.customLogger.append_info(
                f"Episode {self.episode_count} ends, total steps: {self.step_count}, remaining steps: {remaining_steps}"
            )
            self.step_count = 0
        return True
