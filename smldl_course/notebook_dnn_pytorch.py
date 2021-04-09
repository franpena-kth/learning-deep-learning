import time

import torch
import torchsummary
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional
from torchsummary import summary
from torchviz import make_dot
from matplotlib import pyplot


class SequentialNN(nn.Module):

    def __init__(self):
        super(SequentialNN, self).__init__()

        n_input_features = 2
        n_neurons = 3
        # an affine operation: y = Wx + b
        self.output_layer = nn.Linear(n_input_features, n_neurons)

    def forward(self, x):
        x = functional.softmax(self.output_layer(x), dim=1)
        return x


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = functional.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = functional.relu(self.layer1(x))
        x = functional.relu(self.layer2(x))
        x = functional.softmax(self.layer3(x), dim=1)
        return x


def train_network(model, x_train, y_train, x_test, y_test):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    correct = 0
    total = 0
    for epoch in range(1000):

        # Forward pass: Compute predicted y by passing x to the model
        output = model(x_train)

        # Compute and print loss
        loss = criterion(output, y_train)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss.item())
            predict_out = model(x_train)
            _, predict_y = torch.max(predict_out, 1)
            print('Train prediction accuracy', accuracy_score(y_train.data, predict_y.data))

        # if t % 100 == 99:
        #     print(t, loss.item())
        #     # print('y_pred', y_pred.shape)
        #     # print('y', y.shape)
        #     _, predicted = torch.max(y_pred, 1)
        #     # print('predicted', predicted.shape)
        #     total = y.shape[0]
        #     correct = (predicted == y).float().sum().item()
        #     # print('Accuracy', correct)
        #     print('Accuracy: %f %%' % (100 * correct / y.shape[0]))

    predict_out = model(x_test)
    _, predict_y = torch.max(predict_out, 1)
    print('Test prediction accuracy', accuracy_score(y_test.data, predict_y.data))


def pytorch_visualization():
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))

    x = Variable(torch.randn(1, 8))
    y = model(x)

    make_dot(y.mean(), params=dict(model.named_parameters())).render("/tmp/attached", format="png")
    # pyplot.show()


def visualize_network():
    net = SequentialNN()
    print(net)
    summary(net, (1, 2))

    print(net.output_layer)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # output_layer .weight
    print(params[0].data)  # output_layer .weight
    print(net.output_layer.weight)
    print(net.output_layer)

    x = Variable(torch.randn(1, 2))
    y = net(x)
    make_dot(y.mean(), params=dict(net.named_parameters())).render("/tmp/sequential_nn", format="png")


def iris():
    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = iris.target

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)

    print(X.shape)
    print(y.shape)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y.float()).long())
    test_y = Variable(torch.Tensor(test_y.float()).long())

    # model = SequentialNN()
    model = Net()
    train_network(model, train_X, train_y, test_X, test_y)


def main():
    # visualize_network()
    iris()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

# https://janakiev.com/blog/pytorch-iris/
