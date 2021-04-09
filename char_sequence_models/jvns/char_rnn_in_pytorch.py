import random

import numpy
import torch
from torch.nn import functional
from torch import nn
from fastai.text.transform import Vocab
import unidecode
import string


# Taken from https://gist.github.com/jvns/b6dda36b2fdcc02b833ed5b0c7a09112
# Download Hans Christian Anderson's fairy tales
# !wget -O fairy-tales.txt https://www.gutenberg.org/cache/epub/27200/pg27200.txt > /dev/null 2>&1



file = unidecode.unidecode(open('fairy-tales.txt').read())
# Remove the table of contents & Gutenberg preamble
text = file[5000:]
v = Vocab.create((x for x in text), max_vocab=400, min_freq=1)
num_letters = len(v.itos)
# training_set = torch.Tensor(v.numericalize([x for x in text])).type(torch.LongTensor).cuda()
training_set = torch.Tensor(v.numericalize([x for x in text])).type(torch.LongTensor)
training_set = training_set[:100000]


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.hidden = None

    def forward(self, input):
        # input = torch.nn.functional.one_hot(input, num_classes=self.input_size).type(
        #     torch.FloatTensor).cuda().unsqueeze(0)
        input = torch.nn.functional.one_hot(input, num_classes=self.input_size).type(
            torch.FloatTensor).unsqueeze(0)
        if self.hidden is None:
            l_output, self.hidden = self.lstm(input)
        else:
            l_output, self.hidden = self.lstm(input, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        return self.h2o(l_output)


class Trainer():
    def __init__(self, input_size, hidden_size, lr=0.001):
        # self.rnn = MyLSTM(input_size, hidden_size).cuda()
        self.rnn = MyLSTM(input_size, hidden_size)
        # use gradient clipping for some reason
        torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), amsgrad=True, lr=lr)
        self.losses = []
        self.gradients = []
    def epoch(self):
        x = len(self.losses)
        i = 0
        while i < len(training_set) - 40:
            # every time we give it a sequence with a random length (10 to 40)
            # and ask it to predict the next character
            seq_len = random.randint(10, 40)
            input, target = training_set[i:i+seq_len],training_set[i+1:i+1+seq_len]
            i += seq_len
            # forward pass
            output = self.rnn(input)
            loss = functional.cross_entropy(output.squeeze()[-1:], target[-1:])
            # compute gradients and take optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save the losses & gradients so we can graph them if we want
            self.losses.append(loss.item())
            self.gradients.append(torch.norm(self.rnn.lstm.weight_hh_l0.grad))
        # Print the loss for this epoch
        print(numpy.array(self.losses[x:]).mean())
        # Print out some predictions
        print(''.join(make_preds(self.rnn, temperature=1)))


# some helper methods to generate text from the model
def next_pred(rnn, letter, temperature=0.1):
    input = torch.Tensor(v.numericalize([letter])).type(torch.LongTensor).squeeze()
    input = input.unsqueeze(0)
    output = rnn(input)
    prediction_vector = functional.softmax(output.squeeze()/temperature)
    return v.textify(torch.multinomial(prediction_vector, 1).flatten(), sep='').replace('_', ' ')


def make_preds(rnn, n=50, initial='1', temperature=1):
    letter = initial
    for i in range(n):
        letter = next_pred(rnn, letter, temperature=temperature)
        yield letter


trainer = Trainer(input_size=num_letters, hidden_size=150, lr=0.001)


from fastprogress import progress_bar
for i in progress_bar(range(10)):
    trainer.epoch()


print(''.join(make_preds(trainer.rnn, n=3000, temperature=1.1)))


