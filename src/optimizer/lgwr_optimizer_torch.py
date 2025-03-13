from src.model.lgwr_torch import LGWR


class LgwrOptimizer():

    def __init__(self,
                 model: IModel,
                 lbnn_model: LBNN,
                 kernel: LgwrKernel,
                 logger: LgwrLogger,
                 optimizeMode: LgwrOptimizeMode = 'cuda',
                 lr=0.01,
                 epochs=100
                 ):
        super().__init__(model, kernel, logger)

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        self.optimizeMode = optimizeMode

        self.logger.append_info("LGWR Optimizer Initialized")
        self.logger.append_info(
            f"Optimize Mode: {self.optimizeMode}"
        )

        self.lbnn_model = lbnn_model
        self.lbnn_model.to(self.optimizeMode)

        self.optimizer = torch.optim.Adam(
            self.lbnn_model.parameters(),
            lr=lr
        )

        self.epochs = epochs

    def optimize(self):
        self.logger.append_info("Start Optimizing")

        for epoch in range(self.epochs):
            self.logger.append_info(f"Epoch: {epoch}")

            for i, data in enumerate(self.dataset):
                x, y = data

                x = torch.tensor(x, dtype=torch.float32).to(self.optimizeMode)
                y = torch.tensor(y, dtype=torch.float32).to(self.optimizeMode)

                self.optimizer.zero_grad()
                bandwidth = self.lbnn_model(x)
                loss = self.kernel.loss(y, bandwidth)
                loss.backward()
                self.optimizer.step()

                self.logger.append_info(
                    f"Epoch: {epoch}, "
                    f"Loss: {loss.item()}"
                )

        self.logger.append_info("Optimization Finished")

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.optimizeMode)
        bandwidth = self.lbnn_model(x)
        return bandwidth.detach().cpu().numpy()

    def save(self, path):
        torch.save(self.lbnn_model.state_dict(), path)

    def load(self, path):
        self.lbnn_model.load_state_dict(torch.load(path))
        self.lbnn_model.eval()
