import os
import json
from datetime import datetime
from src.log.timestamp import current_time_str
from typing import Literal, TypeAlias


MatricesType: TypeAlias = Literal['AIC',
                                  'AICc',
                                  'R-squared',
                                  'R-squared adjusted']


class ILogger:

    model_type: str = 'baseLogger'
    model_info = {
        'model_type': model_type,
        'bandwidth_optimization': [],
        'info': [],
        'matrices': {
            'AIC': None,
            'AICc': None,
            'R-squared': None,
            'R-squared adjusted': None
        },
    }
    training_process = {
        'aicc_records': None,
        'r2_records': None,
        'bandwidth_mean_records': None,
        'bandwidth_variance_records': None
    }

    def __init__(self):
        self.__init_storage()

    def __init_storage(self):
        self.log_path = os.path.join(
            os.getcwd(), 'logs',  f'{self.model_type}-{current_time_str()}-log')
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)

    def append_info(self, record: str):
        msg = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): record}
        self.model_info['info'].append(
            msg
        )
        print(msg)

    def append_bandwidth_optimization(
        self,
        episode: int,
        reward: float,
        r2: float,
        bw: str | float,
        log: str
    ):
        record = {
            'episode': episode,
            'reward': reward,
            'r2': r2,
            'bandwidth': bw
        }
        self.model_info['bandwidth_optimization'].append(
            record
        )
        msg = {
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"): log
        }
        self.model_info['info'].append(
            msg
        )
        print(msg)
        self.save_model_info_json()

    def append_training_process(self,
                                episode_count: int,
                                aicc_records: list[float],
                                r2_records: list[float],
                                bandwidth_mean_records: list[float] | None,
                                bandwidth_variance_records: list[float] | None
                                ):
        record = {
            'aicc_records': '[' + ', '.join(map(str, aicc_records)) + ']',
            'r2_records': '[' + ', '.join(map(str, r2_records)) + ']',
            'bandwidth_mean_records': '[' + ', '.join(map(str, bandwidth_mean_records or [])) + ']',
            'bandwidth_variance_records': '[' + ', '.join(map(str, bandwidth_variance_records or [])) + ']'
        }
        self.training_process = record
        self.save_training_process(episode_count)

    def update_matrics(self, matrices_type: MatricesType, value: float):
        self.model_info['matrices'][matrices_type] = value

    def save_model_info_json(self):
        with open(os.path.join(self.log_path, 'model_info.json'), 'w') as f:
            json.dump(self.model_info, f, indent=1)

    def save_training_process(self, episode_count: int):
        with open(os.path.join(self.log_path, f'training_process_{episode_count}.json'), 'w') as f:
            json.dump(self.training_process, f, indent=1)
