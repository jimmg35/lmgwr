
from src.kernel.ikernel import IKernel
from src.model.imodel import IModel
from src.log.ilogger import ILogger

from typing import Literal, TypeAlias

GwrBandwidthOptimizeMethod: TypeAlias = Literal['golden_section',
                                                'grid_search', 'random_search']


class IOptimizer:
    model: IModel
    kernel: IKernel
    logger: ILogger

    method: GwrBandwidthOptimizeMethod
    search_range: tuple
