import numpy as np
from collections import OrderedDict


class SGD:
    def __init__(self, params: OrderedDict, lr: float, momentum: float = 0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.velocities = []
        for _, param in self.params.items():
            self.velocities.append(np.zeros(param.weight.shape))

    def step(self):
        for i, (_, param) in enumerate(self.params.items()):
            self.velocities[i] = self.lr * param.grad_weight + self.momentum * self.velocities[i]

            param.weight -= self.velocities[i]
