import time
import math
import torch
import wandb
from torch import nn
from torch.nn import functional

import utils
from unif.unif_data import CodeDescDataset
from unif.unif_evaluate import evaluate_top_n
from unif.unif_model import UNIFAttention
from unif.unif_model_no_attention import UNIFNoAttention


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

    code_token_ids = torch.tensor(tokenized_code['input_ids'], dtype=torch.int64)
    code_token_ids = code_token_ids.reshape(1, -1).to(utils.get_best_device())
    positive_desc_token_ids = tokenized_positive_desc['input_ids']
    positive_desc_token_ids = positive_desc_token_ids.reshape(1, -1).to(utils.get_best_device())
    negative_desc_token_ids = tokenized_negative_desc['input_ids']
    negative_desc_token_ids = negative_desc_token_ids.reshape(1, -1).to(utils.get_best_device())

    code_mask = torch.tensor(tokenized_code['attention_mask'], dtype=torch.int)
    code_mask = code_mask.reshape(1, -1).to(utils.get_best_device())
    positive_desc_mask = tokenized_positive_desc['attention_mask']
    positive_desc_mask = positive_desc_mask.reshape(1, -1).to(utils.get_best_device())
    negative_desc_mask = tokenized_negative_desc['attention_mask']
    negative_desc_mask = negative_desc_mask.reshape(1, -1).to(utils.get_best_device())

    code_embedding, positive_desc_embedding = model(
        code_token_ids, code_mask, positive_desc_token_ids, positive_desc_mask)

    code_embedding, negative_desc_embedding = model(
        code_token_ids, code_mask, negative_desc_token_ids, negative_desc_mask)

    loss = loss_function(code_embedding, positive_desc_embedding, negative_desc_embedding)

    loss.backward()
    optimiser.step()

    return code_embedding, positive_desc_embedding, loss.item()


def train_cycle(use_wandb=True):
    print("%s: Training the model" % (time.strftime("%Y/%m/%d-%H:%M:%S")))

    # n_iters = 100000
    # print_every = 5000
    # plot_every = 1000
    # print_every = 1
    # plot_every = 2
    # embedding_size = 2
    # num_epochs = 300
    print_every = 50
    plot_every = 500
    embedding_size = 32
    num_epochs = 20
    margin = 0.5
    train_size = None
    evaluate_size = 100
    save_path = './unif_model.ckpt'

    # Keep track of losses for plotting
    current_print_loss = 0
    current_plot_loss = 0
    all_losses = []

    start = time.time()
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    dataset = CodeDescDataset(code_snippets_file, descriptions_file, train_size)
    num_iters = len(dataset)
    model = UNIFAttention(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)
    # model = UNIFNoAttention(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)
    model = model.to(utils.get_best_device())
    cosine_similarity_function = nn.CosineSimilarity()

    # loss_function = nn.CosineEmbeddingLoss()
    loss_function = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - functional.cosine_similarity(x, y), margin=margin)
    loss_function = loss_function.to(utils.get_best_device())
    learning_rate = 0.05  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_wandb:
        name = model.__class__.__name__.lower() + '-triplet-cosine'
        wandb.init(project='code-search', name=name, reinit=True)
        config = wandb.config
        config.learning_rate = learning_rate
        config.embedding_size = embedding_size
        config.evaluate_size = evaluate_size
        config.margin = margin
        config.num_epochs = num_epochs
        config.train_size = len(dataset)
        wandb.watch(model, log_freq=plot_every)
        metrics = evaluate_top_n(model, evaluate_size)
        if use_wandb:
            wandb.log(metrics)

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)

        for iter in range(num_iters):
            # print(iter)
            tokenized_code, tokenized_positive_desc, tokenized_negative_desc =\
                dataset[iter]
            code_embedding, desc_embedding, loss = train(
                model, loss_function, optimiser, tokenized_code,
                tokenized_positive_desc, tokenized_negative_desc)
            current_print_loss += loss
            current_plot_loss += loss

            # Print iter number, loss, name and guess
            if (iter + 1) % print_every == 0:
                print('%d %d%% (%s) %.4f' % (iter + 1, (iter + 1) / num_iters * 100, timeSince(start), current_print_loss / print_every))
                cosine_similarity = cosine_similarity_function(code_embedding, desc_embedding).item()
                print('Cosine similarity:', cosine_similarity)
                # print('Cosine similarity:', cosine_similarity, code_embedding, desc_embedding)
                current_print_loss = 0

            # Add current loss avg to list of losses
            if (iter + 1) % plot_every == 0:
                torch.save(model.state_dict(), save_path)
                metrics = evaluate_top_n(model, evaluate_size)
                metrics.update({'loss': current_plot_loss / plot_every})
                all_losses.append(current_plot_loss / plot_every)
                current_plot_loss = 0
                if use_wandb:
                    wandb.log(metrics)

    return model, current_print_loss, all_losses
