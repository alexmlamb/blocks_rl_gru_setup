from .a2c import A2cAgent
from ..config import Config


class AcktrAgent(A2cAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.precond = config.preconditioner(self.net)

    def _step_optimizer(self) -> None:
        """Approximates F^-1∇h and apply it.
        """
        self.precond.step()
        self.optimizer.step()
