
from typing import Literal, TypeAlias

# cuda(GPU) or cpu(CPU)
OptimizeMode: TypeAlias = Literal[
    'cuda', 'cpu'
]
