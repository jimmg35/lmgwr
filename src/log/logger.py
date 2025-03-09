import os
import json
import logging
from datetime import datetime
from src.log.timestamp import current_time_str
from typing import Literal, TypeAlias


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


GwrMatricesType: TypeAlias = Literal['AIC',
                                     'AICc', 'R-squared', 'R-squared adjusted']


class GwrLogger:
    model_type: str

    def __init__(self):

        self.model_type = 'gwr'
        self.log_path = os.path.join(
            os.getcwd(), 'logs',  f'{self.model_type}-{current_time_str()}-log')
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)

        self.model_info = {
            'model_type': self.model_type,
            'info': [],
            'matrices': {
                'AIC': None,
                'AICc': None,
                'R-squared': None,
                'R-squared adjusted': None
            },
            'bandwidth_optimization': []
        }

    def append_info(self, record: str):
        msg = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): record}
        self.model_info['info'].append(
            msg
        )
        print(msg)

    def append_bandwidth_optimization(self, record: str):
        msg = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): record}
        self.model_info['bandwidth_optimization'].append(
            msg
        )
        print(msg)

    def update_matrics(self, matrices_type: GwrMatricesType, value: float):
        self.model_info['matrices'][matrices_type] = value

    def save_model_info_json(self):
        with open(os.path.join(self.log_path, 'model_info.json'), 'w') as f:
            json.dump(self.model_info,
                      f,
                      indent=1,
                      #   separators=(',', ': ')
                      )


class LgwrLogger(GwrLogger):

    def __init__(self):
        self.model_type = 'lgwr'
        self.log_path = os.path.join(
            os.getcwd(), 'logs',  f'{self.model_type}-{current_time_str()}-log')
        if os.path.exists(self.log_path) is False:
            os.makedirs(self.log_path)

        self.model_info = {
            'model_type': self.model_type,
            'info': [],
            'matrices': {
                'AIC': None,
                'AICc': None,
                'R-squared': None,
                'R-squared adjusted': None
            },
            # 'bandwidth_optimization': []
        }
