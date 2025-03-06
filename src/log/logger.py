import os
import logging
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
