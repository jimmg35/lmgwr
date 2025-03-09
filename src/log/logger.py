import os
import json
import logging
from datetime import datetime
from src.log.timestamp import current_time_str


def create_logger():
    log_path = os.path.join(os.getcwd(), 'logs',  f'{current_time_str()}-log')
    if os.path.exists(log_path) is False:
        os.makedirs(log_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(
                                os.path.join(log_path, 'record.log')),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)


class GwrLogger:
    def __init__(self):
        self.model_info = {
            'info': [],
            'bandwidth_optimization': [],
            'matrices': {
                'AIC': None,
                'AICc': None,
                'R-squared': None,
                'R-squared adjusted': None
            }
        }

    def append_info(self, record: str):
        self.model_info['info'].append(
            f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {record}"
        )

    def append_bandwidth_optimization(self, record: str):
        self.model_info['bandwidth_optimization'].append(
            f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {record}"
        )

    def save_model_info_json(self):
        log_path = os.path.join(os.getcwd(), 'logs',
                                f'{current_time_str()}-log')
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)

        with open(os.path.join(log_path, 'model_info.json'), 'w') as f:
            json.dump(self.model_info, f)
