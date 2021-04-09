
from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        rnn_output, self.hidden = self.rnn_layer(input, self.hidden)
        output = self.output_layer(rnn_output)
        output = output[-1]  # We output the last element of the sequence
        output = self.softmax(output)
        return output

    def initHidden(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)
