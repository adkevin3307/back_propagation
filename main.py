import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Union

import nn
from Model import Model


class Net:
    def __init__(self, criterion: Union[nn.MSE, nn.CrossEntropy]):
        self.criterion = criterion

        self.linear_1 = nn.Linear(in_features=2, out_features=16)
        self.linear_2 = nn.Linear(in_features=16, out_features=32)
        self.linear_3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.linear_1.forward(x)
        x = nn.Sigmoid.forward(x)
        self.layer_1 = x

        x = self.linear_2.forward(x)
        x = nn.Sigmoid.forward(x)
        self.layer_2 = x

        x = self.linear_3.forward(x)
        x = nn.Sigmoid.forward(x)
        self.layer_3 = x

        return x

    def backward(self, input: np.ndarray, target: np.ndarray) -> None:
        dx = self.criterion.backward(input, target)

        dx = np.multiply(dx, nn.Sigmoid.backward(self.layer_3))
        dx = self.linear_3.backward(dx)

        dx = np.multiply(dx, nn.Sigmoid.backward(self.layer_2))
        dx = self.linear_2.backward(dx)

        dx = np.multiply(dx, nn.Sigmoid.backward(self.layer_1))
        dx = self.linear_1.backward(dx)

    def update(self, lr: float) -> None:
        self.linear_1.update(lr)
        self.linear_2.update(lr)
        self.linear_3.update(lr)


def generate_linear(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    inputs, labels = [], []

    for pt in np.random.uniform(0, 1, (n, 2)):
        inputs.append(pt)
        labels.append(0 if pt[0] > pt[1] else 1)

    return (np.array(inputs), np.array(labels).reshape(n, 1))


def generate_XOR_easy() -> tuple[np.ndarray, np.ndarray]:
    inputs, labels = [], []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if i == 5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return (np.array(inputs), np.array(labels).reshape(21, 1))


def show_history(history: dict) -> None:
    _, axes = plt.subplots()

    axes.plot(history['loss'], color='blue')
    axes.set_ylabel('Loss', color='blue', fontsize=14)

    axes = axes.twinx()

    axes.plot(history['accuracy'], color='red')
    axes.set_ylabel('Accuracy', color='red', fontsize=14)

    plt.show()


def show_result(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)

    for i in range(len(x)):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)

    for i in range(len(x)):
        if y_hat[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='linear')
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)

    args = parser.parse_args()

    if args.dataset == 'linear':
        X_train, Y_train = generate_linear(n=1000)
        X_test, Y_test = generate_linear(n=100)
    elif args.dataset == 'xor':
        X_train, Y_train = generate_XOR_easy()
        X_test, Y_test = generate_XOR_easy()
    else:
        raise RuntimeError('Dataset Not Found')

    criterion = nn.CrossEntropy()
    net = Net(criterion=criterion)

    model = Model(net)
    train_history = model.train(X_train, Y_train, epochs=args.epochs, lr=args.lr)
    test_history = model.test(X_test, Y_test)

    show_history(train_history)
    show_result(X_test, Y_test, test_history['predict'])
