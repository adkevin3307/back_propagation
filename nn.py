import numpy as np
from typing import Any, Union
from collections import OrderedDict

import grad


class Autograd:
    layers = []

    @staticmethod
    def push(layer: dict[str, Any]) -> None:
        Autograd.layers.append(layer)

    @staticmethod
    def pop() -> Union[dict[str, Any], None]:
        if Autograd.popable():
            return Autograd.layers.pop()

        return None

    @staticmethod
    def popable() -> bool:
        return (len(Autograd.layers) > 0)


class Module:
    def __init__(self) -> None:
        self._parameters = OrderedDict()

    def __setattr__(self, name: str, value: Union[np.ndarray, 'Module']) -> None:
        params = self.__dict__.get('_parameters')

        if params == None:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
            self._parameters[name] = value

    def parameters(self) -> OrderedDict:
        return self._parameters

    def forward(self) -> None:
        raise NotImplementedError('Forward Not Implemented')


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = np.random.normal(size=(in_features, out_features))
        self.grad_weight = np.zeros((in_features, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = np.dot(x, self.weight)

        if grad.gradable():
            layer = {}
            layer['x_prev'] = np.array(x, copy=True)
            layer['backward'] = self.backward

            Autograd.push(layer)

        return self.z

    def backward(self, grad_prev: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        self.grad_weight = np.dot(x_prev.T, grad_prev)

        return np.dot(grad_prev, self.weight.T)


class Sigmoid:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x = 1.0 / (1.0 + np.exp(-x))

        if grad.gradable():
            layer = {}
            layer['x_prev'] = np.array(x, copy=True)
            layer['backward'] = Sigmoid.backward

            Autograd.push(layer)

        return x

    @staticmethod
    def backward(grad_prev: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        return grad_prev * np.multiply(x_prev, 1.0 - x_prev)


class ReLU:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x = np.maximum(0, x)

        if grad.gradable():
            layer = {}
            layer['x_prev'] = np.array(x, copy=True)
            layer['backward'] = ReLU.backward

            Autograd.push(layer)

        return x

    @staticmethod
    def backward(grad_prev: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        dx = np.array(grad_prev, copy=True)
        dx[x_prev == 0] = 0

        return dx


class MSE:
    @staticmethod
    def forward(input: np.ndarray, target: np.ndarray) -> float:
        return np.sum(np.square(input - target)) / input.shape[0]

    @staticmethod
    def backward(input: np.ndarray, target: np.ndarray) -> None:
        dx = 2.0 * (input - target) / input.shape[0]

        while Autograd.popable():
            layer = Autograd.pop()
            dx = layer['backward'](dx, layer['x_prev'])


class CrossEntropy:
    @staticmethod
    def forward(input: np.ndarray, target: np.ndarray) -> float:
        return -1.0 * np.sum(target * np.log(input) + (1.0 - target) * np.log(1.0 - input)) / input.shape[0]

    @staticmethod
    def backward(input: np.ndarray, target: np.ndarray) -> None:
        dx = ((1 - target) / (1 - input) - (target / input)) / input.shape[0]

        while Autograd.popable():
            layer = Autograd.pop()
            dx = layer['backward'](dx, layer['x_prev'])
