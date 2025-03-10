import os
import json
import logging
from datetime import datetime
from typing import Literal, TypeAlias

from src.log.timestamp import current_time_str
from src.log.ilogger import ILogger


GwrMatricesType: TypeAlias = Literal['AIC',
                                     'AICc', 'R-squared', 'R-squared adjusted']


class GwrLogger(ILogger):
    model_type: str = 'gwr'
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
        super().__init__()


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
