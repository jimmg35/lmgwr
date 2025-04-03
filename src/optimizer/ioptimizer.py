from src.kernel.ikernel import IKernel
from src.model.imodel import IModel
from src.log.ilogger import ILogger


class IOptimizer:
    model: IModel
    kernel: IKernel
    logger: ILogger

    search_range: tuple

    def __init__(self,
                 model: IModel,
                 kernel: IKernel,
                 logger: ILogger) -> None:
        self.model = model
        self.kernel = kernel
        self.logger = logger

    def optimize(self):
        raise NotImplementedError("Method not implemented")

    # def objective_function(self, bandwidth: float) -> float:
    #     raise NotImplementedError("Method not implemented")
