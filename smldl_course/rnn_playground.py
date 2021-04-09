import numpy
import seaborn
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn


num_epochs = 20
learning_rate = 0.01


# https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_layer_size, batch_size, output_size=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_layer_size, output_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


def train_network(model, X_train, y_train):
    loss_fn = torch.nn.MSELoss(size_average=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #####################
    # Train model
    #####################

    hist = numpy.zeros(num_epochs)

    for t in range(num_epochs):
        # Clear stored gradient
        model.zero_grad()

        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()

        # Forward pass
        y_pred = model(X_train)

        loss = loss_fn(y_pred, y_train)
        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

#####################################################################################
#####################################################################################
#####################################################################################


# https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
class CoronaVirusPredictor(nn.Module):

    def __init__(self, input_size, hidden_layer_size, seq_len, num_layers=2):

        super(CoronaVirusPredictor, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers, dropout=0.5)

        # Define the output layer
        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=1)

    def reset_hidden_state(self):

        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_layer_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_layer_size)
        )

    def forward(self, sequences):

        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.hidden_layer_size)[-1]
        y_pred = self.linear(last_time_step)

        return y_pred


def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 60
    train_hist = numpy.zeros(num_epochs)
    test_hist = numpy.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(train_data)
        loss = loss_fn(y_pred.float(), train_labels)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), test_labels)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')

        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model.eval(), train_hist, test_hist


#####################################################################################
#####################################################################################
#####################################################################################


# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
class LSTM_2(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # print('input_seq', input_seq.shape)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_lstm(model, train_inout_seq):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


def evaluate_lstm(model):
    fut_pred = 12
    train_window = 12

    test_data_size = 12

    flight_data = seaborn.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)
    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(
        numpy.array(test_inputs[train_window:]).reshape(-1, 1))




#####################################################################################
#####################################################################################
#####################################################################################


class MyOwnRNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, sequence_length, output_size, num_layers):

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.num_layers = num_layers

        # Create the RNN layer
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers)

        # Create the output layer
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_layer_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_layer_size)
        )

    def forward(self, input_sequence):

        reshaped_sequence = input_sequence.sequence.view(len(input_sequence), self.sequence_length, -1)
        rnn_output, self.hidden = self.rnn_layer(reshaped_sequence, self.hidden)

        predictions = self.linear(rnn_output[-1].view(self.batch_size, -1))
        print('Predictions 1', predictions.shape)
        # return predictions.view(-1)
        predictions = self.linear(rnn_output.view(self.seq_len, len(input_sequence), self.hidden_layer_size)[-1])
        print('Predictions 2', predictions.shape)
        # return predictions
        predictions = self.linear(rnn_output.view(len(input_sequence), -1))
        print('Predictions 3', predictions.shape)
        # return predictions[-1]

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
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


# def train_network(model, X_train, y_train):
#     loss_fn = torch.nn.MSELoss(size_average=False)
#     optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     for t in range(num_epochs):
#         model.zero_grad()
#         model.hidden = model.init_hidden()
#         y_pred = model(X_train)
#         loss = loss_fn(y_pred, y_train)
#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()
#
# def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     for t in range(num_epochs):
#         model.reset_hidden_state()
#         y_pred = model(train_data)
#         loss = loss_fn(y_pred.float(), train_labels)
#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()
#
# def train_lstm(model, train_inout_seq):
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     for i in range(num_epochs):
#         for seq, labels in train_inout_seq:
#             optimizer.zero_grad()
#             model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                                  torch.zeros(1, 1, model.hidden_layer_size))
#
#             y_pred = model(seq)
#             loss = loss_function(y_pred, labels)
#             loss.backward()
#             optimizer.step()


def main():
    fut_pred = 12
    train_window = 12

    test_data_size = 12

    flight_data = seaborn.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    print(len(train_inout_seq))
    print(train_inout_seq[0])
    print(len(train_inout_seq[0][0]))
    print(len(train_inout_seq[0][1]))


main()
