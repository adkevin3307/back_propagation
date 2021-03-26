import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='linear')
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-m', '--momentum', type=float, default=0.0)

    args = parser.parse_args()

    return args


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
