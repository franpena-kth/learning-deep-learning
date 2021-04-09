import math
import time

import torch
from torch import nn

from nlp_from_scratch.generating_names.gn_data import DataProcessor
from nlp_from_scratch.generating_names.gn_model import RNN


def train(model, criterion, optimiser, category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()

    model.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimiser.step()

    return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_cycle(data_processor):

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()
    model = RNN(data_processor.n_letters, 128, data_processor.n_letters, data_processor.n_categories)
    criterion = nn.NLLLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor =\
            data_processor.randomTrainingExample()
        output, loss = train(model, criterion, optimiser, category_tensor, input_line_tensor, target_line_tensor)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    return model, total_loss, all_losses
