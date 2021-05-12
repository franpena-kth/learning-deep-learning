import time

import torch
from torch import nn
import numpy

from unif.top_n_evaluator import prepare_embeddings_for_topn_evaluation, compute_metrics, output_scores
from unif.unif_data import CodeDescDataset
from unif.unif_tokenizer import tokenize_data


def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count += 1
    return count / len(results)


def evaluate_top_n(model, size=None):
    print("%s: Performing Top-N evaluation" % (time.strftime("%Y/%m/%d-%H:%M:%S")))

    code_snippets_file = './data/parallel_bodies_n1000'
    descriptions_file = './data/parallel_desc_n1000'
    test_dataset = CodeDescDataset(code_snippets_file, descriptions_file, size)

    tokenized_code_data, code_mask, tokenized_desc_data, desc_mask = tokenize_data(test_dataset)
    tokenized_code_data = tokenized_code_data.to(torch.int64)
    tokenized_desc_data = tokenized_desc_data.to(torch.int64)
    print('Tokenized code data', tokenized_code_data.shape)
    print('Tokenized desc data', tokenized_desc_data.shape)

    code_embedding_data, desc_embedding_data = model(
        tokenized_code_data, code_mask, tokenized_desc_data, desc_mask)
    print('Code embedding data', code_embedding_data.shape)
    print('Desc embedding data', desc_embedding_data.shape)

    with torch.no_grad():
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        results = {}

        for rowid, desc_embedding in enumerate(desc_embedding_data):
            # Calculate the cosine similarity between the code and desc embeddings
            code_desc_similarity = cosine_similarity(code_embedding_data[rowid].reshape((1, -1)),
                                                     desc_embedding.reshape((1, -1)))

            other_code_embeddings = numpy.delete(code_embedding_data, rowid, 0)
            tiled_desc = torch.Tensor(numpy.tile(desc_embedding, (other_code_embeddings.shape[0], 1)))

            # print('Other + tiled', other_code_embeddings.shape, tiled_desc.shape)

            # Calculate the cosine similarity between the description vector and all the code snippets excepting the code that matches the desc
            ress = cosine_similarity(other_code_embeddings, tiled_desc)
            results[rowid] = len(ress[ress >= code_desc_similarity])

        top_1 = get_top_n(1, results)
        top_3 = get_top_n(3, results)
        top_5 = get_top_n(5, results)
        top_15 = get_top_n(15, results)

        print('Top 1', top_1)
        print('Top 3', top_3)
        print('Top 5', top_5)
        print('Top 15', top_15)

        queries_ids, relevant_docs, queries_result_list =\
            prepare_embeddings_for_topn_evaluation(code_embedding_data, desc_embedding_data)
        metrics = compute_metrics(queries_result_list, queries_ids, relevant_docs)
        output_scores(metrics)
        metrics.update({'Top_1': top_1, 'Top_3': top_3, 'Top_5': top_5, 'Top_15': top_15})

        return metrics
