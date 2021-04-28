
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


def train(
        model, loss_function, optimiser, tokenized_code,
        tokenized_positive_desc, tokenized_negative_desc):
    model.train()
    model.zero_grad()

    code_token_ids = torch.tensor(tokenized_code['input_ids'], dtype=torch.int)
    code_token_ids = code_token_ids.reshape(1, -1)
    positive_desc_token_ids = tokenized_positive_desc['input_ids']
    positive_desc_token_ids = positive_desc_token_ids.reshape(1, -1)
    negative_desc_token_ids = tokenized_negative_desc['input_ids']
    negative_desc_token_ids = negative_desc_token_ids.reshape(1, -1)

    code_mask = torch.tensor(tokenized_code['attention_mask'], dtype=torch.int)
    code_mask = code_mask.reshape(1, -1)
    positive_desc_mask = tokenized_positive_desc['attention_mask']
    positive_desc_mask = positive_desc_mask.reshape(1, -1)
    negative_desc_mask = tokenized_negative_desc['attention_mask']
    negative_desc_mask = negative_desc_mask.reshape(1, -1)

    POSITIVE_SIMILARITY = torch.ones(1)
    code_embedding, positive_desc_embedding = model(
        code_token_ids, code_mask, positive_desc_token_ids, positive_desc_mask)
    positive_loss = loss_function(code_embedding, positive_desc_embedding, POSITIVE_SIMILARITY)

    NEGATIVE_SIMILARITY = -torch.ones(1)
    code_embedding, negative_desc_embedding = model(
        code_token_ids, code_mask, negative_desc_token_ids, negative_desc_mask)
    negative_loss = loss_function(code_embedding, negative_desc_embedding, NEGATIVE_SIMILARITY)

    total_loss = positive_loss + negative_loss
    total_loss.backward()
    optimiser.step()

    return code_embedding, positive_desc_embedding, total_loss.item(), positive_loss.item(), negative_loss.item()


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

    loss_function = nn.CosineEmbeddingLoss(margin=0.05)
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
            tokenized_code, tokenized_positive_desc, tokenized_negative_desc = dataset[iter]
            code_embedding, desc_embedding, loss, positive_loss, negative_loss = train(
                model, loss_function, optimiser,
                tokenized_code, tokenized_positive_desc, tokenized_negative_desc)
            current_loss += loss

            # Print iter number, loss, name and guess
            if (iter + 1) % print_every == 0:
                print('%d %d%% (%s) %.4f' % (iter + 1, (iter + 1) / num_iters * 100, timeSince(start), current_loss / print_every))

            # Print iter number, loss, name and guess
            if (iter + 1) % print_top_n_every == 0:
                torch.save(model.state_dict(), save_path)
                metrics = evaluate_top_n(model, evaluate_size)
                metrics.update({'loss': loss})
                metrics.update({'positive_loss': positive_loss})
                metrics.update({'negative_loss': negative_loss})
                wandb.log(metrics)

            # Add current loss avg to list of losses
            if (iter + 1) % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
        # metrics = evaluate_top_n(model, evaluate_size)
        # metrics.update({'loss': loss})
        # wandb.log(metrics)

    return model, current_loss, all_losses

