
from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        combined = torch.cat((input, self.hidden), 1)
        self.hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output

    def initHidden(self):
        self.hidden = torch.zeros(1, self.hidden_size)
