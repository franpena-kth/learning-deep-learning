
import time
import math
import torch
import wandb
from torch import nn

from unif.unif_data import CodeDescDataset
from unif.unif_evaluate import evaluate_top_n
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
    print_top_n_every = 5000
    plot_every = 500
    embedding_size = 128
    num_epochs = 10
    train_size = None
    evaluate_size = 100
    save_path = './unif_model.ckpt'

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    dataset = CodeDescDataset(code_snippets_file, descriptions_file, train_size)
    num_iters = len(dataset)
    wandb.init(project='code-search', name='unif', reinit=True)
    model = UNIF(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)

    loss_function = nn.CosineEmbeddingLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    config = wandb.config
    config.learning_rate = learning_rate
    config.embedding_size = embedding_size
    config.evaluate_size = evaluate_size
    config.train_size = train_size
    wandb.watch(model)

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)

        for iter in range(num_iters):
            # print(iter)
            tokenized_code, tokenized_desc = dataset[iter]
            code_embedding, desc_embedding, loss = train(model, loss_function, optimiser, tokenized_code, tokenized_desc)
            current_loss += loss

            # Print iter number, loss, name and guess
            if (iter + 1) % print_every == 0:
                print('%d %d%% (%s) %.4f' % (iter + 1, (iter + 1) / num_iters * 100, timeSince(start), current_loss / print_every))

            # Print iter number, loss, name and guess
            if (iter + 1) % print_top_n_every == 0:
                torch.save(model.state_dict(), save_path)
                metrics = evaluate_top_n(model, evaluate_size)
                wandb.log(metrics)

            # Add current loss avg to list of losses
            if (iter + 1) % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
        metrics = evaluate_top_n(model, evaluate_size)
        wandb.log(metrics)

    return model, current_loss, all_losses

