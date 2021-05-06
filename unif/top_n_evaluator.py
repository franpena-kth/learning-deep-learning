
import logging
import time
from typing import List

import numpy as np
import torch
import tqdm
from torch import nn


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def compute_metrics(
        queries_result_list: List[object], queries_ids, relevant_docs,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10, 100],
        precision_recall_at_k: List[int] = [1, 3, 5, 10, 100],
        map_at_k: List[int] = [100],
                    # accuracy_at_k,
                    # precision_recall_at_k, mrr_at_k, ndcg_at_k, map_at_k
        ):
    # Init score computation values
    num_hits_at_k = {k: 0 for k in accuracy_at_k}
    precisions_at_k = {k: [] for k in precision_recall_at_k}
    recall_at_k = {k: [] for k in precision_recall_at_k}
    MRR = {k: 0 for k in mrr_at_k}
    ndcg = {k: [] for k in ndcg_at_k}
    AveP_at_k = {k: [] for k in map_at_k}

    # Compute scores on results
    for query_itr in range(len(queries_result_list)):
        query_id = queries_ids[query_itr]

        # Sort scores
        top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
        query_relevant_docs = relevant_docs[query_id]

        # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
        for k_val in accuracy_at_k:
            for hit in top_hits[0:k_val]:
                if hit['corpus_id'] in query_relevant_docs:
                    num_hits_at_k[k_val] += 1
                    break

        # Precision and Recall@k
        for k_val in precision_recall_at_k:
            num_correct = 0
            for hit in top_hits[0:k_val]:
                if hit['corpus_id'] in query_relevant_docs:
                    num_correct += 1

            precisions_at_k[k_val].append(num_correct / k_val)
            recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

        # MRR@k
        for k_val in mrr_at_k:
            for rank, hit in enumerate(top_hits[0:k_val]):
                if hit['corpus_id'] in query_relevant_docs:
                    MRR[k_val] += 1.0 / (rank + 1)
                    break

        # NDCG@k
        for k_val in ndcg_at_k:
            predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in
                                   top_hits[0:k_val]]
            true_relevances = [1] * len(query_relevant_docs)

            ndcg_value = compute_dcg_at_k(predicted_relevance, k_val) / compute_dcg_at_k(true_relevances,
                                                                                                   k_val)
            ndcg[k_val].append(ndcg_value)

        # MAP@k
        for k_val in map_at_k:
            num_correct = 0
            sum_precisions = 0

            for rank, hit in enumerate(top_hits[0:k_val]):
                if hit['corpus_id'] in query_relevant_docs:
                    num_correct += 1
                    sum_precisions += num_correct / (rank + 1)

            avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
            AveP_at_k[k_val].append(avg_precision)

    # Compute averages
    for k in num_hits_at_k:
        num_hits_at_k[k] /= len(queries_ids)

    for k in precisions_at_k:
        precisions_at_k[k] = np.mean(precisions_at_k[k])

    for k in recall_at_k:
        recall_at_k[k] = np.mean(recall_at_k[k])

    for k in ndcg:
        ndcg[k] = np.mean(ndcg[k])

    for k in MRR:
        MRR[k] /= len(queries_ids)

    for k in AveP_at_k:
        AveP_at_k[k] = np.mean(AveP_at_k[k])

    return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg,
            'mrr@k': MRR, 'map@k': AveP_at_k}


def output_scores(scores):
    for k in scores['accuracy@k']:
        logger.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k] * 100))

    for k in scores['precision@k']:
        logger.info("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k] * 100))

    for k in scores['recall@k']:
        logger.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))

    for k in scores['mrr@k']:
        logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

    for k in scores['ndcg@k']:
        logger.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))

    for k in scores['map@k']:
        logger.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))


def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg


def test_case_1():
    queries_result_list = [
        [{'corpus_id': 'd1', 'score': 0.2}, {'corpus_id': 'd2', 'score': 0.5}, {'corpus_id': 'd3', 'score': 0.8}, {'corpus_id': 'd4', 'score': 0.8}]
    ]
    queries_ids = ['q1']
    relevant_docs = {'q1': {'d1'}}

    metrics = compute_metrics(queries_result_list, queries_ids, relevant_docs)
    output_scores(metrics)


def test_case_2():
    queries_result_list = [
        [{'corpus_id': 'd1', 'score': 0.2}, {'corpus_id': 'd2', 'score': 0.5}, {'corpus_id': 'd3', 'score': 0.8}, {'corpus_id': 'd4', 'score': 0.8}],
        [{'corpus_id': 'd1', 'score': 0.2}, {'corpus_id': 'd2', 'score': 0.5}, {'corpus_id': 'd3', 'score': 0.8}, {'corpus_id': 'd4', 'score': 0.8}]
    ]
    queries_ids = ['q1', 'q2']
    relevant_docs = {
        'q1': {'d1'},
        'q2': {'d2'}
    }

    metrics = compute_metrics(queries_result_list, queries_ids, relevant_docs)
    output_scores(metrics)


def prepare_embeddings_for_topn_evaluation(code_embeddings, desc_embeddings):

    assert len(code_embeddings) == len(desc_embeddings),\
        'code_embeddings and desc_embeddings must have the same length'

    num_pairs = len(code_embeddings)
    num_digits = len(str(num_pairs))

    code_ids = ['c' + str(i).zfill(num_digits) for i in range(num_pairs)]
    desc_ids = ['d' + str(i).zfill(num_digits) for i in range(num_pairs)]

    relevant_codes = {}
    for code_id, desc_id in zip(code_ids, desc_ids):
        relevant_codes[desc_id] = {code_id}

    with torch.no_grad():
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        descriptions_result_list = [[] for _ in range(num_pairs)]

        for i in range(num_pairs):
            # query_id = query_ids[i]
            for j in range(num_pairs):
                code_id = 'c' + str(j).zfill(num_digits)
                # score = cosine_similarity(code_embeddings[i], desc_embeddings[j])
                score = cosine_similarity(code_embeddings[i].reshape((1, -1)), desc_embeddings[j].reshape((1, -1))).item()
                descriptions_result_list[i].append({'corpus_id': code_id, 'score': score})

    # queries_result_list = [
    #     [{'corpus_id': 'd1', 'score': 0.2}, {'corpus_id': 'd2', 'score': 0.5}, {'corpus_id': 'd3', 'score': 0.8},
    #      {'corpus_id': 'd4', 'score': 0.8}],
    #     [{'corpus_id': 'd1', 'score': 0.2}, {'corpus_id': 'd2', 'score': 0.5}, {'corpus_id': 'd3', 'score': 0.8},
    #      {'corpus_id': 'd4', 'score': 0.8}]
    # ]
    # queries_ids = ['q1', 'q2']
    # relevant_docs = {
    #     'q1': {'d1'},
    #     'q2': {'d2'}
    # }

    return desc_ids, relevant_codes, descriptions_result_list


def main():

    # test_case_1()
    code_embedding = torch.rand(3, 2)
    desc_embedding = torch.rand(3, 2)
    desc_ids, relevant_codes, descriptions_result_list =\
        prepare_embeddings_for_topn_evaluation(code_embedding, desc_embedding)

    print('Description IDs', desc_ids)
    print('Relevant codes', relevant_codes)
    print('Descriptions result list', descriptions_result_list)

    metrics = compute_metrics(descriptions_result_list, desc_ids, relevant_codes)
    output_scores(metrics)

    # test_case_2()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

