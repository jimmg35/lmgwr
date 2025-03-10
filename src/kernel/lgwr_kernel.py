from src.kernel.gwr_kernel import GwrKernel


class LgwrKernel(GwrKernel):
    def __init__(self, dataset, bandwidth, kernel_name):
        super().__init__(dataset, bandwidth, kernel_name)
