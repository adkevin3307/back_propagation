import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = np.random.normal(size=(in_features, out_features))
        self.grad_weight = np.zeros((in_features, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_prev = x
        self.z = np.dot(x, self.weight)

        return self.z

    def backward(self, grad_prev: np.ndarray) -> np.ndarray:
        self.grad_weight = np.dot(self.x_prev.T, grad_prev)

        return np.dot(grad_prev, self.weight.T)

    def update(self, learning_rate: float) -> None:
        self.weight -= learning_rate * self.grad_weight


class Sigmoid:
    x_prev = []

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        Sigmoid.x_prev.append(1.0 / (1.0 + np.exp(-x)))

        return Sigmoid.x_prev[-1]

    @staticmethod
    def backward(grad_prev: np.ndarray) -> np.ndarray:
        x_prev = Sigmoid.x_prev.pop()

        return grad_prev * np.multiply(x_prev, 1.0 - x_prev)


class ReLU:
    x_prev = []

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        ReLU.x_prev.append(np.maximum(0, x))

        return ReLU.x_prev[-1]

    @staticmethod
    def backward(grad_prev: np.ndarray) -> np.ndarray:
        x_prev = ReLU.x_prev.pop()

        dx = np.array(grad_prev, copy=True)
        dx[x_prev == 0] = 0

        return dx


class MSE:
    @staticmethod
    def forward(input: np.ndarray, target: np.ndarray) -> float:
        return np.sum(np.square(input - target)) / len(input)

    @staticmethod
    def backward(input: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2.0 * (input - target) / len(input)


class CrossEntropy:
    @staticmethod
    def forward(input: np.ndarray, target: np.ndarray) -> float:
        return -1.0 * np.sum(target * np.log(input) + (1.0 - target) * np.log(1.0 - input)) / len(input)

    @staticmethod
    def backward(input: np.ndarray, target: np.ndarray) -> np.ndarray:
        return ((1 - target) / (1 - input) - (target / input)) / len(input)
