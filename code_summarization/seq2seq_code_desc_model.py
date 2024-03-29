import io
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, get_tokenizer
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.vocab import Vocab

import utils
from aladdin_persson.aladdin_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# spacy_ger = spacy.load('de_core_news_md')
# spacy_eng = spacy.load('en_core_web_sm')
import de_core_news_md
import en_core_web_sm


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        # target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


def prepare_data():
    spacy_ger = de_core_news_md.load()
    spacy_eng = en_core_web_sm.load()

    def tokenize_ger(text):
        return [tok.text for tok in spacy_ger.tokenizer(text)]

    def tokenize_eng(text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]

    german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

    english = Field(
        tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
    )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english)
    )

    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)

    return train_data, valid_data, test_data, german, english


def train():
    train_data, valid_data, test_data, german, english = prepare_data()

    ### We're ready to define everything we need for training our Seq2Seq model ###

    # Training hyperparameters
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 64

    # Model hyperparameters
    load_model = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = len(german.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024  # Needs to be the same for both RNN's
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5

    # Tensorboard to get nice loss plot
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    encoder_net = Encoder(
        input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
    ).to(device)

    decoder_net = Decoder(
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net, len(english.vocab), device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint_2_2.pth.tar"), model, optimizer)


    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    for epoch in range(num_epochs):
        print(f"{time.strftime('%Y/%m/%d-%H:%M:%S')}: [Epoch {epoch} / {num_epochs}]")

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename='seq2seq_code_desc_model_checkpoint.pth.tar')

        model.eval()

        translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )

        print(f"Translated example sentence: \n {translated_sentence}")

        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target)

            # print('Input', inp_data.shape)
            # print('Target', target.shape)
            # print('Output', output.shape)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            # print("Training loss", loss)
            step += 1

    score = bleu(test_data[1:100], model, german, english, device)
    print(f"Bleu score {score*100:.2f}")


def main():

    train()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
