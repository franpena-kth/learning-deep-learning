import time

import utils
from unif.unif_data import CodeDescDataset
from unif.unif_evaluate import evaluate_top_n
from unif.unif_plots import plot
from unif.unif_random_model import RandomModel
from unif.unif_tokenizer import tokenize_data
from unif.unif_train import train_cycle


def test_random_model():
    random_model = RandomModel(embedding_size=128)
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    test_dataset = CodeDescDataset(code_snippets_file, descriptions_file, 100)

    tokenized_code_data, code_mask, tokenized_desc_data, desc_mask = tokenize_data(test_dataset)
    print('Tokenized code data', tokenized_code_data.shape)
    print('Tokenized desc data', tokenized_desc_data.shape)

    code_embedding_data, desc_embedding_data = random_model(tokenized_code_data, code_mask, tokenized_desc_data, desc_mask)
    print('Code embedding data', code_embedding_data.shape)
    print('Desc embedding data', desc_embedding_data.shape)

    evaluate_top_n(code_embedding_data, desc_embedding_data)


def test_unif_model():
    unif_model, current_loss, all_losses = train_cycle()
    plot(all_losses)
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    test_dataset = CodeDescDataset(code_snippets_file, descriptions_file, 100)

    tokenized_code_data, code_mask, tokenized_desc_data, desc_mask = tokenize_data(test_dataset)
    print('Tokenized code data', tokenized_code_data.shape)
    print('Tokenized desc data', tokenized_desc_data.shape)

    code_embedding_data, desc_embedding_data = unif_model(tokenized_code_data, code_mask, tokenized_desc_data, desc_mask)
    print('Code embedding data', code_embedding_data.shape)
    print('Desc embedding data', desc_embedding_data.shape)

    evaluate_top_n(code_embedding_data, desc_embedding_data)


def main():
    utils.plant_seeds()
    # test_unif_model()
    test_random_model()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
