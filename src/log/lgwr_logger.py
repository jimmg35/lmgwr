
from src.log.gwr_logger import GwrLogger


class LgwrLogger(GwrLogger):
    model_type = 'lgwr'

    def __init__(self):
        super().__init__()

        self.model_info = {
            'model_type': self.model_type,
            'matrices': {
                'AIC': None,
                'AICc': None,
                'R-squared': None,
                'R-squared adjusted': None
            },
            'info': [],
            'bandwidth_optimization': []
        }
