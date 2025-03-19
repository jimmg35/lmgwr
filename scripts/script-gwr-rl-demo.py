
from stable_baselines3 import PPO
import pandas as pd

from src.optimizer.reinforce.gwr_optimizer import GwrOptimizerRL
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.optimizer.reinforce.callback import EpisodeTracker
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger
from src.model.gwr import GWR

if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        georgia_data,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        logger,
        isSpherical=True
    )

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(
        spatialDataset,
        logger,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    gwr = GWR(spatialDataset, kernel, logger)

    # Initial gwr gym environment
    env = GwrOptimizerRL(
        gwr,
        logger,
        min_bandwidth=10,
        max_bandwidth=spatialDataset.x_matrix.shape[0],
        min_action=-10,
        max_action=10,
        max_steps=500,
        reward_threshold=0.75
    )

    # Using PPO to optimize the bandwidth
    TOTAL_TIMESTEPS = 5000
    episodeTracker = EpisodeTracker(
        logger,
        total_timesteps=TOTAL_TIMESTEPS
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu'
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=episodeTracker
    )
    logger.append_info("PPO: PPO finished training.")

    # Test the model
    obs, _ = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        logger.append_info(
            f"Bandwidth: {obs}, Reward (R2): {reward}"
        )
        if done or truncated:
            break

    # Save the log
    logger.save_model_info_json()
