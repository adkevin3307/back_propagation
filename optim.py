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


class Adagrad:
    def __init__(self, params: OrderedDict, lr: float = 0.01, eps: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.eps = eps

        self.sums = []
        for _, param in self.params.items():
            self.sums.append(np.zeros(param.weight.shape))

    def step(self):
        for i, (_, param) in enumerate(self.params.items()):
            self.sums[i] += np.square(param.grad_weight)
            adaptive_lr = self.lr * (1.0 / (np.sqrt(self.sums[i]) + self.eps))

            param.weight -= adaptive_lr * param.grad_weight
