import numpy as np

import nn
import optim
from Model import Model
from utils import parse, generate_linear, generate_XOR_easy, show_history, show_result


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=16)
        self.linear_2 = nn.Linear(in_features=16, out_features=32)
        self.linear_3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.linear_1.forward(x)
        x = nn.Sigmoid.forward(x)

        x = self.linear_2.forward(x)
        x = nn.Sigmoid.forward(x)

        x = self.linear_3.forward(x)
        x = nn.Sigmoid.forward(x)

        return x


if __name__ == '__main__':
    np.random.seed(0)

    args = parse()

    if args.dataset == 'linear':
        X_train, Y_train = generate_linear(n=100)
        X_test, Y_test = generate_linear(n=100)
    elif args.dataset == 'xor':
        X_train, Y_train = generate_XOR_easy()
        X_test, Y_test = generate_XOR_easy()
    else:
        raise RuntimeError('Dataset Not Found')

    net = Net()

    if args.criterion == 'mse':
        criterion = nn.MSE()
    elif args.criterion == 'crossentropy':
        criterion = nn.CrossEntropy()
    else:
        raise RuntimeError('Criterion Not Found')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    model = Model(net, criterion, optimizer)
    train_history = model.train(X_train, Y_train, epochs=args.epochs)
    test_history = model.test(X_test, Y_test)

    show_history(train_history)
    show_result(X_test, Y_test, test_history['predict'])
