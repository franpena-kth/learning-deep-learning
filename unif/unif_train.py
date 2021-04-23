
import time
import math
import torch
from torch import nn

from unif.unif_data import CodeDescDataset
from unif.unif_model import UNIF


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(model, loss_function, optimiser, tokenized_code, tokenized_desc):
    model.zero_grad()

    code_token_ids = torch.tensor(tokenized_code['input_ids'], dtype=torch.int)
    code_token_ids = code_token_ids.reshape(1, -1)
    desc_token_ids = tokenized_desc['input_ids']
    desc_token_ids = desc_token_ids.reshape(1, -1)

    code_mask = torch.tensor(tokenized_code['attention_mask'], dtype=torch.int)
    code_mask = code_mask.reshape(1, -1)
    desc_mask = tokenized_desc['attention_mask']
    desc_mask = desc_mask.reshape(1, -1)

    code_embedding, desc_embedding = model(code_token_ids, code_mask, desc_token_ids, desc_mask)

    POSITIVE_SIMILARITY = torch.ones(1)
    NEGATIVE_SIMILARITY = -torch.ones(1)
    loss = loss_function(code_embedding, desc_embedding, POSITIVE_SIMILARITY)
    loss.backward()
    optimiser.step()

    return code_embedding, desc_embedding, loss.item()


def train_cycle():
    print("%s: Training the model" % (time.strftime("%Y/%m/%d-%H:%M:%S")))

    # n_iters = 100000
    # print_every = 5000
    # plot_every = 1000
    print_every = 500
    plot_every = 500
    embedding_size = 128

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    dataset = CodeDescDataset(code_snippets_file, descriptions_file)
    n_iters = len(dataset)
    n_hidden = 128
    model = UNIF(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)

    loss_function = nn.CosineEmbeddingLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for iter in range(n_iters):
        # print(iter)
        tokenized_code, tokenized_desc = dataset[iter]
        code_embedding, desc_embedding, loss = train(model, loss_function, optimiser, tokenized_code, tokenized_desc)
        current_loss += loss

        # Print iter number, loss, name and guess
        if (iter + 1) % print_every == 0:
            print('%d %d%% (%s) %.4f' % (iter + 1, (iter + 1) / n_iters * 100, timeSince(start), current_loss / print_every))

        # Add current loss avg to list of losses
        if (iter + 1) % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return model, current_loss, all_losses

