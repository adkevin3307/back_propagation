import numpy as np


class SGD:
    def __init__(self, params: np.ndarray, lr: float, momentum: float = 0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.velocities = []
        for param in self.params:
            self.velocities.append(np.zeros(param.weight.shape))

    def step(self):
        for i, param in enumerate(self.params):
            self.velocities[i] = self.lr * param.grad_weight + self.momentum * self.velocities[i]

            param.weight -= self.velocities[i]
