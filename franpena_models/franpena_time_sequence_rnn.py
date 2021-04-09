import time

import torch
from torch import nn

from franpena_models.franpena_charts import plot_series, plot_true_vs_predicted
from franpena_models.franpena_datasets import create_time_series_train_test_sets, \
    create_airline_passengers_train_test_sets, create_covid_train_test_sets,\
    create_flights_train_test_sets, create_sine_train_test_sets

dataset_settings = {
    'hands_on_time_series': {
        'num_epochs': 50,
        'sequence_length': 50,
        'learning_rate': 0.01,
        'dataset_creator': create_time_series_train_test_sets
    },
    'airplane_passengers': {
        'num_epochs': 2000,
        'sequence_length': 4,
        'learning_rate': 0.01,
        'dataset_creator': create_airline_passengers_train_test_sets
    },
    'covid': {
        'num_epochs': 600,
        'sequence_length': 5,
        'learning_rate': 0.01,
        'dataset_creator': create_covid_train_test_sets
    },
    'flights': {
        'num_epochs': 1000,
        'sequence_length': 12,
        'learning_rate': 0.01,
        'dataset_creator': create_flights_train_test_sets
    },
    'sine': {
        'num_epochs': 1000,
        'sequence_length': 12,
        'learning_rate': 0.01,
        'dataset_creator': create_sine_train_test_sets
    }
}

# dataset = 'hands_on_time_series'
dataset = 'airplane_passengers'
# dataset = 'covid'
# dataset = 'flights'
# dataset = 'sine'

N_EPOCHS = dataset_settings[dataset]['num_epochs']
SEQUENCE_LENGTH = dataset_settings[dataset]['sequence_length']
LEARNING_RATE = dataset_settings[dataset]['learning_rate']
dataset_creator = dataset_settings[dataset]['dataset_creator']


class FranpenaTimeSequenceRNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(FranpenaTimeSequenceRNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Create the RNN layer
        self.rnn_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_layer_size,
            num_layers=num_layers, batch_first=True)

        # Create the output layer
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def reset_hidden_state(self, batch_size):
        # Use this for LSTM which has a hidden state and a cell state
        self.hidden = (
            # n_layers * n_directions, batch_size, rnn_hidden_size
            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size)
        )

        # Use this for RNN and GRU
        # self.hidden = torch.zeros(
        #   self.num_layers, batch_size, self.hidden_layer_size)

    def forward(self, input_sequence):
        rnn_output, self.hidden = self.rnn_layer(input_sequence, self.hidden)
        output = self.output_layer(rnn_output)

        # We return the value for the last time step
        return output[:, -1, :]


def train(model, X, y):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        # Zero gradients and reset hidden state
        optimizer.zero_grad()
        model.zero_grad()
        model.reset_hidden_state(X.size(0))

        # Make predictions and calculate the loss
        output = model(X)
        loss = loss_function(output, y)

        # Execute back propagation and calculate the gradients
        loss.backward()

        # Update the parameters
        optimizer.step()

        print(f'Epoch: {epoch:3} - Loss: {loss.item():10.10f}')


def evaluate(model, X, y, scaler, train_size):
    model.eval()
    with torch.no_grad():
        loss_function = nn.MSELoss()
        model.reset_hidden_state(X.size(0))
        predicted_test = model(X)
        loss = loss_function(predicted_test, y)
        print(f'Test set loss: {loss.item():10.10f}')
        plot_series(
            SEQUENCE_LENGTH, X[0, :, 0], y[0, 0], predicted_test[0, 0])

        plot_true_vs_predicted(y, predicted_test, train_size, scaler)


def main():
    # X_full, y_full, X_train, y_train, X_test, y_test, scaler = create_time_series_train_test_sets(SEQUENCE_LENGTH)
    # X_full, y_full, X_train, y_train, X_test, y_test, scaler = create_airline_passengers_train_test_sets(SEQUENCE_LENGTH)
    X_full, y_full, X_train, y_train, X_test, y_test, scaler =\
        dataset_creator(SEQUENCE_LENGTH)

    dataX = X_full
    dataY = y_full
    train_size = X_train.size(0)

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print('X_full', X_full.shape)
    print('y_full', y_full.shape)

    # make_plots(X_test, y_test)

    model = FranpenaTimeSequenceRNN(
        input_size=1, hidden_layer_size=100, output_size=1, num_layers=1)
    train(model, X_train, y_train)
    evaluate(model, dataX, dataY, scaler, train_size)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
