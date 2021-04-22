import time

from unif.unif_data import CodeDescDataset
from unif.unif_evaluate import evaluate_top_n
from unif.unif_plots import plot
from unif.unif_tokenizer import tokenize_data
from unif.unif_train import train_cycle


def main():
    unif_model, current_loss, all_losses = train_cycle()
    plot(all_losses)
    code_snippets_file = './data/parallel_bodies'
    descriptions_file = './data/parallel_desc'
    dataset = CodeDescDataset(code_snippets_file, descriptions_file)

    tokenized_code_data, tokenized_desc_data = tokenize_data(dataset)
    print('Tokenized code data', tokenized_code_data.shape)
    print('Tokenized desc data', tokenized_desc_data.shape)

    code_embedding_data, desc_embedding_data = unif_model(tokenized_code_data, tokenized_desc_data)
    print('Code embedding data', code_embedding_data.shape)
    print('Desc embedding data', desc_embedding_data.shape)

    evaluate_top_n(code_embedding_data, desc_embedding_data)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
