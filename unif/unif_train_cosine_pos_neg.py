
import time
import math
import torch
import wandb
from torch import nn

from unif.unif_data import CodeDescDataset
from unif.unif_evaluate import evaluate_top_n
from unif.unif_model import UNIFAttention
from unif.unif_model_no_attention import UNIFNoAttention


"""
Created a UNIF variant that uses both positive cosine and negative cosine
similarity for positive and negative code-description pairs, respectively
"""


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(
        model, positive_loss_function, negative_loss_function, optimiser,
        tokenized_code, tokenized_positive_desc, tokenized_negative_desc):
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
    positive_loss = positive_loss_function(
        code_embedding, positive_desc_embedding, POSITIVE_SIMILARITY)

    NEGATIVE_SIMILARITY = -torch.ones(1)
    code_embedding, negative_desc_embedding = model(
        code_token_ids, code_mask, negative_desc_token_ids, negative_desc_mask)
    negative_loss = negative_loss_function(
        code_embedding, negative_desc_embedding, NEGATIVE_SIMILARITY)

    total_loss = positive_loss + negative_loss
    total_loss.backward()
    optimiser.step()

    return code_embedding, positive_desc_embedding, total_loss.item(), positive_loss.item(), negative_loss.item()


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
    num_epochs = 500
    positive_margin = 0.0
    negative_margin = 0.5
    train_size = None
    evaluate_size = 100
    save_path = './unif_model.ckpt'

    # Keep track of losses for plotting
    current_print_loss = 0
    current_plot_loss = 0
    all_losses = []

    start = time.time()
    code_snippets_file = './data/parallel_bodies_n1000'
    descriptions_file = './data/parallel_desc_n1000'
    dataset = CodeDescDataset(code_snippets_file, descriptions_file, train_size)
    num_iters = len(dataset)
    # model = UNIF(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)
    model = UNIFNoAttention(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)
    cosine_similarity_function = nn.CosineSimilarity()

    positive_loss_function = nn.CosineEmbeddingLoss(margin=positive_margin)
    negative_loss_function = nn.CosineEmbeddingLoss(margin=negative_margin)
    learning_rate = 0.05  # If you set this too high, it might explode. If too low, it might not learn
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if use_wandb:
        wandb.init(project='code-search', name='unif-cosine-pos-neg', reinit=True)
        config = wandb.config
        config.learning_rate = learning_rate
        config.embedding_size = embedding_size
        config.evaluate_size = evaluate_size
        config.positive_margin = positive_margin
        config.negative_margin = negative_margin
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
            code_embedding, desc_embedding, loss, positive_loss, negative_loss = train(
                model, positive_loss_function, negative_loss_function, optimiser,
                tokenized_code, tokenized_positive_desc, tokenized_negative_desc)
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
                metrics.update({'loss': loss})
                metrics.update({'current_loss': current_plot_loss / plot_every})
                metrics.update({'positive_loss': positive_loss})
                metrics.update({'negative_loss': negative_loss})
                all_losses.append(current_plot_loss / plot_every)
                current_plot_loss = 0
                if use_wandb:
                    wandb.log(metrics)

    return model, current_print_loss, all_losses
