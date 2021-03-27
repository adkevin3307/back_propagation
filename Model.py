import numpy as np

import grad


class Model:
    def __init__(self, net, criterion, optimizer) -> None:
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int) -> dict:
        history = {'accuracy': [], 'loss': []}

        length = len(str(epochs))
        train_loader = list(zip(X_train, Y_train))

        for epoch in range(epochs):
            correct = 0
            total_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                x, y = np.array(x).reshape(1, -1), np.array(y)

                y_hat = self.net(x)

                correct += np.sum(np.around(y_hat).astype(np.int32) == y)

                loss = self.criterion.forward(y_hat, y)
                total_loss += loss

                self.criterion.backward(y_hat, y)
                self.optimizer.step()

                current_progress = (i + 1) / len(train_loader) * 100
                progress_bar = '=' * int((i + 1) * (20 / len(train_loader)))
                print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, loss: {loss:.3f}', end='')

            total_loss /= len(train_loader)
            accuracy = correct / len(train_loader)

            print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{"=" * 20}], ', end='')
            print(f'loss: {total_loss:.3f}, accuracy: {accuracy:.3f}')

            history['loss'].append(total_loss)
            history['accuracy'].append(accuracy)

        return history

    def test(self, X_test: np.ndarray, Y_test: np.ndarray, is_test=True) -> dict:
        history = {}

        test_loader = list(zip(X_test, Y_test))

        predict = []
        correct = 0
        total_loss = 0.0
        with grad.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = np.array(x).reshape(1, -1), np.array(y)

                y_hat = self.net.forward(x)
                predict.append(y_hat)

                correct += np.sum(np.around(y_hat).astype(np.int32) == y)

                loss = self.criterion.forward(y_hat, y)
                total_loss += loss

                current_progress = (i + 1) / len(test_loader) * 100
                progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))
                print(f'\rTest: [{progress_bar:<20}] {current_progress:6.2f}%, loss: {loss:.3f}', end='')

        total_loss /= len(test_loader)
        accuracy = correct / len(test_loader)

        history['predict'] = np.around(np.array(predict)).astype(np.int32)
        history['loss'] = total_loss
        history['accuracy'] = accuracy

        prefix = 'test' if is_test == True else 'val'
        print(f'\rTest: [{"=" * 20}], ', end='')
        print(f'{prefix}_loss: {total_loss:>.3f}, {prefix}_accuracy: {accuracy:.3f}')

        return history
