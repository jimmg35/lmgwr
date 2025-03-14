import os
import json
import logging
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
        'matrices': {
            'AIC': None,
            'AICc': None,
            'R-squared': None,
            'R-squared adjusted': None
        },
        'info': [],
        'bandwidth_optimization': []
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

    def append_bandwidth_optimization(self, record: str):
        msg = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): record}
        self.model_info['bandwidth_optimization'].append(
            msg
        )
        print(msg)

    def update_matrics(self, matrices_type: MatricesType, value: float):
        self.model_info['matrices'][matrices_type] = value

    def save_model_info_json(self):
        with open(os.path.join(self.log_path, 'model_info.json'), 'w') as f:
            json.dump(self.model_info,
                      f,
                      indent=1,
                      #   separators=(',', ': ')
                      )
