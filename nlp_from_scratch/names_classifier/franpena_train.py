import math
import time
import torch
from torch import nn

from nlp_from_scratch.names_classifier.franpena_data import DataProcessor, NamesDataset
from nlp_from_scratch.names_classifier.franpena_model import RNN


def train(model, category_tensor, line_tensor):
    model.initHidden()
    criterion = nn.NLLLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output = model(line_tensor[i])

    loss = criterion(output, category_tensor)
    loss.backward()
    optimiser.step()

    # loss.backward()
    # # Add parameters' gradients to their values, multiplied by learning rate
    # for p in model.parameters():
    #     p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_cycle():
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()
    data_processor = DataProcessor()
    # data_set = OwnDataset()
    n_hidden = 128
    model = RNN(data_processor.n_letters, n_hidden, data_processor.n_categories)

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = \
            data_processor.randomTrainingExample()
    # for iter in range(1, n_iters + 1):
    #     category, line, category_tensor, line_tensor =\
    #         data_set[iter]
        output, loss = train(model, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = data_processor.categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return model, current_loss, all_losses

