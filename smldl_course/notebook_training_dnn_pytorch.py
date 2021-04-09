import math
import time

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional
import torchvision
from matplotlib import pyplot
from torchvision import transforms
from six.moves import urllib

N_EPOCHS = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
EARLY_STOP_THRESHOLD = 0.01


def init():

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)


def load_mnist():
    # Uncomment this only when you have not downloaded the dataset
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    classes = range(10)

    # show_images(testloader)

    return trainloader, testloader, classes


def show_images(testloader):
    examples = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = pyplot.figure()
    for i in range(6):
        pyplot.subplot(2, 3, i + 1)
        pyplot.tight_layout()
        pyplot.imshow(example_data[i][0], cmap='gray', interpolation='none')
        pyplot.title("Ground Truth: {}".format(example_targets[i]))
        pyplot.xticks([])
        pyplot.yticks([])
    fig.show()


class BiDenseNetwork(nn.Module):

    def __init__(self):
        super(BiDenseNetwork, self).__init__()
        input_size = 784
        hidden_layer_size = 128
        # hidden_layer_2_size = 64
        output_size = 10
        self.flatten = nn.Flatten(1, -1)
        self.dense_layer = nn.Linear(input_size, hidden_layer_size)
        torch.nn.init.kaiming_uniform_(self.dense_layer.weight)  # He initialization
        # self.dropout_layer = nn.Dropout(0.2)
        # self.batch_normalization_layer = nn.BatchNorm1d(hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        torch.nn.init.kaiming_uniform_(self.dense_layer.weight)  # He initialization

    def forward(self, x):
        x = self.flatten(x)
        # print('x', x.shape)
        x = self.dense_layer(x)
        # x = self.dropout_layer(x)
        # x = self.batch_normalization_layer(x)
        x = functional.relu(x)
        x = self.output_layer(x)
        x = functional.softmax(x, dim=1)

        return x


def train_network(model, train_loader, test_loader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.9)

    last_loss = math.inf
    for epoch in range(N_EPOCHS):

        running_loss = 0
        running_accuracy = 0
        for images, labels in train_loader:
            output = model(images)
            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predict_out = model(images)
            _, predict_y = torch.max(predict_out, 1)
            running_accuracy += accuracy_score(labels.data, predict_y.data)
        # print('Train prediction accuracy', accuracy_score(labels.data, predict_y.data))
        print(f'{time.strftime("%Y/%m/%d-%H:%M:%S")} - '
              f'Epoch {epoch} - '
              f'Training loss: {running_loss / len(train_loader)} - '
              f'Training accuracy: {running_accuracy / len(train_loader)}')

        loss_delta = (last_loss / running_loss) - 1
        last_loss = running_loss
        print(f'Loss delta: {loss_delta}')
        if loss_delta < EARLY_STOP_THRESHOLD:
            print('Early stopping!')
            return

    # predict_out = model(x_test)
    # _, predict_y = torch.max(predict_out, 1)
    # print('Test prediction accuracy', accuracy_score(y_test.data, predict_y.data))


def test_network(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


def main():
    init()
    trainloader, testloader, classes = load_mnist()
    model = BiDenseNetwork()
    print(model)
    train_network(model, trainloader, testloader)
    test_network(model, testloader)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))


# https://github.com/pytorch/examples/tree/master/mnist
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
