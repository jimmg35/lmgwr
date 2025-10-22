from src.log.gwr_logger import GwrLogger


class LmgwrLogger(GwrLogger):
    model_type = 'lmgwr'

    def __init__(self):
        super().__init__()
        self.model_info = {
            'model_type': self.model_type,
            'bandwidth_optimization': [],
            'info': [],
            'matrices': {
                'AIC': None,
                'AICc': None,
                'R-squared': None,
                'R-squared adjusted': None
            }
        }

