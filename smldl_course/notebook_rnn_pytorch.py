import time

import numpy as np

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.autograd import Variable

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


N_STEPS = 50
N_EPOCHS = 20
LEARNING_RATE = 0.005


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)


def load_data():
    np.random.seed(42)

    series = generate_time_series(10000, N_STEPS + 1)
    X_train, y_train = series[:7000, :N_STEPS], series[:7000, -1]
    X_test, y_test = series[7000:, :N_STEPS], series[7000:, -1]

    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(X_train).float())
    test_X = Variable(torch.Tensor(X_test).float())
    train_y = Variable(torch.Tensor(y_train).float())
    test_y = Variable(torch.Tensor(y_test).float())

    return train_X, train_y, test_X, test_y


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):

    plt.plot(series, ".-")
    if y is not None:
        plt.plot(N_STEPS, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(N_STEPS, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, N_STEPS + 1, -1, 1])


def plot(X_test, y_test):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    for col in range(3):
        plt.sca(axes[col])
        plot_series(X_test[col, :, 0], y_test[col, 0], y_label=("$x(t)$" if col == 0 else None))
    plt.show()


class FullyConnectedNetwork(nn.Module):

    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()

        self.flatten = nn.Flatten(1, -1)
        self.dense_layer = nn.Linear(N_STEPS, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_layer(x)

        return x


class SimpleRnnNetwork(nn.Module):

    def __init__(self):
        super(SimpleRnnNetwork, self).__init__()

        # self.rnn = nn.RNN(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.hidden_layer_size = 1

        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_layer_size)

        self.linear = nn.Linear(self.hidden_layer_size, out_features=1)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # print('Input seq', input_seq.shape)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_network(model, x_train, y_train, x_test, y_test):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(N_EPOCHS):
        # Forward pass: Compute predicted y by passing x to the model
        output = model(x_train)

        # Compute and print loss
        loss = loss_function(output, y_train)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'{time.strftime("%Y/%m/%d-%H:%M:%S")} - '
              f'Epoch {epoch} - Training loss: {loss} - ')


def train_sequence_network(model, x_train, y_train, x_test, y_test):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print('xtrain', x_train.shape)
    print('ytrain', y_train.shape)

    for epoch in range(N_EPOCHS):
        # Forward pass: Compute predicted y by passing x to the model
        output, hidden = model(x_train)

        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        print(output.shape)
        print(y_train.shape)

        # Compute and print loss
        loss = loss_function(output, y_train)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'{time.strftime("%Y/%m/%d-%H:%M:%S")} - '
              f'Epoch {epoch} - Training loss: {loss} - ')


def train_sequence_network2(model, x_train, y_train, x_test, y_test):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for i in range(N_EPOCHS):
        for seq, labels in zip(x_train, y_train):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        # if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


def test_network(model, x_test, y_test):

    with torch.no_grad():
        outputs = model(x_test)
        loss = mean_squared_error(y_test, outputs)

        print('MSE of the network on the test set: %f' % loss)

    y_pred = model(x_test)
    x_test = x_test.detach().numpy()
    y_test = y_test.detach().numpy()
    y_pred = y_pred.detach().numpy()
    plot_series(x_test[0, :, 0], y_test[0, 0], y_pred[0, 0])
    plt.show()


#####################################################################################
#####################################################################################
#####################################################################################


class MyOwnRNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, sequence_length, output_size, num_layers):

        super(MyOwnRNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.num_layers = num_layers

        # Create the RNN layer
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers)

        # Create the output layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.sequence_length, self.hidden_layer_size),
            torch.zeros(self.num_layers, self.sequence_length, self.hidden_layer_size)
        )

    def forward(self, input_sequence):

        reshaped_sequence = input_sequence.view(len(input_sequence), self.sequence_length, -1)
        # print('Reshape sequence', reshaped_sequence.shape)
        # print(self.hidden)
        # rnn_output, self.hidden = self.rnn_layer(reshaped_sequence, self.hidden)
        rnn_output, self.hidden = self.rnn_layer(reshaped_sequence)

        # predictions = self.linear(rnn_output[-1].view(self.sequence_length, -1))
        # print('Predictions 1', predictions.shape, predictions.view(-1).shape)
        # return predictions.view(-1)
        predictions = self.linear(rnn_output.view(self.sequence_length, len(input_sequence), self.hidden_layer_size)[-1])
        # print('Predictions 2', predictions.shape, predictions.shape)
        # return predictions
        # predictions = self.linear(rnn_output.view(len(input_sequence), -1))
        # print('Predictions 3', predictions.shape)
        # return predictions[-1]
        # lstm_out, self.hidden = self.rnn_layer(input_sequence.view(len(input_sequence), self.sequence_length, -1))
        # predictions = self.linear(lstm_out.view(len(input_sequence), -1))
        # print('Predictions 3', predictions.shape)

        return predictions[-1]

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        # lstm_out, self.hidden = self.lstm(input_sequence.view(len(input_sequence), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        # return y_pred.view(-1)

        # lstm_out, self.hidden = self.lstm(
        #     sequences.view(len(sequences), self.seq_len, -1),
        #     self.hidden
        # )
        # last_time_step = lstm_out.view(self.seq_len, len(sequences), self.hidden_layer_size)[-1]
        # y_pred = self.linear(last_time_step)
        #
        # return y_pred

        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # return predictions[-1]


def my_own_train(model, x_train, y_train):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        # Zero gradients and reset hidden state
        model.zero_grad()
        optimizer.zero_grad()
        model.reset_hidden_state()

        # Make predictions and calculate the loss
        output = model(x_train)
        loss = loss_function(output, y_train)

        # Execute back propagation and calculate the gradients
        loss.backward()

        # Update the parameters
        optimizer.step()

        print(f'Epoch: {epoch:3} - Loss: {loss.item():10.10f}')


def see_dataset_shapes():
    X_train, y_train, X_test, y_test = load_data()

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)


def main():
    X_train, y_train, X_test, y_test = load_data()
    # plot(X_test, y_test)
    print('X_train', X_train.shape)
    # model = FullyConnectedNetwork()
    # model = SimpleRnnNetwork()
    # train_network(model, X_train, y_train, X_test, y_test)
    # train_sequence_network(model, X_train, y_train, X_test, y_test)
    # train_sequence_network2(model, X_train, y_train, X_test, y_test)
    # test_network(model, X_test, y_test)
    # model = MyOwnRNN(input_size=1, hidden_layer_size=100, sequence_length=50, output_size=1, num_layers=1)
    # my_own_train(model, X_train, y_train)

    see_dataset_shapes()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

