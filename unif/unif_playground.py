import random
import time

import torch

import utils
from unif.unif_data import CodeDescDataset
from unif.unif_model import UNIFAttention
from unif.unif_tokenizer import tokenize_data


def load_unif_model():
    load_path = './unif_model.ckpt'
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    train_size = 11
    embedding_size = 128
    dataset = CodeDescDataset(code_snippets_file, descriptions_file, train_size)
    model = UNIFAttention(dataset.code_vocab_size, dataset.desc_vocab_size, embedding_size)
    model.load_state_dict(torch.load(load_path))

    code_snippet = dataset.code_snippets[3]
    description = dataset.descriptions[3]

    # code_snippet_10 = dataset.code_snippets[10]
    # description_10 = dataset.descriptions[10]

    print(code_snippet)
    print(description)
    # print()
    # print(code_snippet_10)
    # print(description_10)
    # print()

    tokenized_code_data, code_mask, tokenized_desc_data, desc_mask =\
        tokenize_data(dataset)
    code_embedding, desc_embedding = model(
        tokenized_code_data, code_mask, tokenized_desc_data, desc_mask)

    print(code_embedding[10])
    print(desc_embedding[10])


def main():
    # load_unif_model()
    print(utils.get_best_device())


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
