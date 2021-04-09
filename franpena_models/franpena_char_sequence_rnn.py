import time

import numpy
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional
from torch.utils.data import DataLoader

from franpena_models.franpena_datasets import create_char_dataset, CharDataset, create_word_dataset


LEARNING_RATE = 0.002
N_EPOCHS = 100
SEQUENCE_LENGTH = 20


class FranpenaCharSequenceRNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(FranpenaCharSequenceRNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Create the embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=input_size, embedding_dim=input_size)

        # Create the RNN layer
        self.rnn_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_layer_size,
            num_layers=num_layers)

        # Create the output layer
        self.decoder = nn.Linear(hidden_layer_size, output_size)

    def reset_hidden_state(self, batch_size):
        # Use this for LSTM which has a hidden state and a cell state
        self.hidden = (
            # n_layers * n_directions, batch_size, rnn_hidden_size
            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size)
        )

    def forward(self, input_sequence):

        # print('input_sequence', input_sequence.shape)

        embedding = self.embedding_layer(input_sequence)
        rnn_output, hidden_state = self.rnn_layer(embedding, self.hidden)
        self.hidden = (hidden_state[0].detach(), hidden_state[1].detach())
        self.hidden = (hidden_state[0].detach(), hidden_state[1].detach())
        decoder_output = self.decoder(rnn_output)

        return decoder_output


def train(model, data):

    data_size = len(data)
    # save_path = "./FranpenaCharRNN_test.pth"
    save_path = "./FranpenaWordRNN_source.pth"

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for i_epoch in range(1, N_EPOCHS + 1):

        # random starting point (1st 100 chars) from data to begin
        data_ptr = numpy.random.randint(100)
        n = 0
        running_loss = 0
        model.hidden = None

        # Zero gradients and reset hidden state
        optimizer.zero_grad()
        model.reset_hidden_state(data.size(1))

        # This loop runs until the end of the dataset
        while True:
            input_seq = data[data_ptr: data_ptr + SEQUENCE_LENGTH]
            target_seq = data[data_ptr + 1: data_ptr + SEQUENCE_LENGTH + 1]

            # print('input_seq', input_seq.shape, 'target_seq', target_seq.shape)

            # forward pass
            # output, hidden_state = model(input_seq, hidden_state)
            output = model(input_seq)

            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            # print('outupt', output.shape, 'target_seq', target_seq.shape,
            #       'ouput squezeed', torch.squeeze(output).shape, 'target_seq_squeeze',
            #       torch.squeeze(target_seq).shape)


            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the data pointer
            data_ptr += SEQUENCE_LENGTH
            n += 1

            # if at end of data : break
            if data_ptr + SEQUENCE_LENGTH + 1 > data_size:
                break

        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss / n))
        torch.save(model.state_dict(), save_path)


def train_with_data_loader(model, dataset):

    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(N_EPOCHS):

        data_pointer = numpy.random.randint(100)
        n = 0
        running_loss = 0
        model.hidden = None

        # Reset the hidden state
        model.reset_hidden_state(1)

        for batch, (x, y) in enumerate(dataloader):

            x = x.squeeze(dim=0)
            y = y.squeeze(dim=0)
            # print('x', x.shape, 'y', y.shape)

            # Zero gradients
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred.transpose(1, 2), y)
            # loss = criterion(y_pred, y)
            running_loss += loss.item()

            # print('x', x.shape, 'y', y.shape, 'y_pred', y_pred.shape)

            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += 1

        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch, running_loss / n))
        # torch.save(model.state_dict(), save_path)




def evaluate(model, data, ix_to_char):

    data_size = len(data)
    output_seq_len = 200  # total num of characters in output test sequence

    # sample / generate a text sequence after every epoch
    data_ptr = 0
    hidden_state = None

    # random character from data to begin
    rand_index = numpy.random.randint(data_size - 1)
    input_seq = data[rand_index: rand_index + 1]

    print("----------------------------------------")
    while True:
        # forward pass
        # output, hidden_state = model(input_seq, hidden_state)
        output = model(input_seq)

        # construct categorical distribution and sample a character
        output = functional.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample()

        # print the sampled character
        print(ix_to_char[index.item()], end=' ')

        # next input is current output
        input_seq[0][0] = index.item()
        data_ptr += 1

        if data_ptr > output_seq_len:
            break

    print("\n----------------------------------------")


def main():
    # set random seed to 0
    numpy.random.seed(0)
    torch.manual_seed(0)

    # data, char_to_ix, ix_to_char, data_size, vocab_size = create_char_dataset()
    data, char_to_ix, ix_to_char, data_size, vocab_size = create_word_dataset()

    # print('X_train', X_train.shape)
    # print('y_train', y_train.shape)
    # print('X_test', X_test.shape)
    # print('y_test', y_test.shape)
    # print('X_full', X_full.shape)
    # print('y_full', y_full.shape)

    # make_plots(X_test, y_test)

    model = FranpenaCharSequenceRNN(
        input_size=vocab_size, hidden_layer_size=16, output_size=vocab_size,
        num_layers=1)
    train(model, data)
    # dataset = CharDataset(SEQUENCE_LENGTH)
    # train_with_data_loader(model, dataset)
    evaluate(model, data, ix_to_char)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

