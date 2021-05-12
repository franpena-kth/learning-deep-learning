"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use the negative
passages (the triplets) that are provided by the MS MARCO dataset.

Running this script:
python train_bi-encoder.py
"""
import time

import wandb
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import os
from collections import defaultdict
from torch.utils.data import IterableDataset


#### Just some code to print debug information to stdout
from sbert.WandbInformationRetrievalEvaluator import WandbInformationRetrievalEvaluator
from sbert.sbert_data import CodeDescRawTripletDataset
from unif.unif_data import CodeDescDataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# The  model we want to fine-tune
# model_name = 'distilroberta-base'
model_name = 'sshleifer/tiny-distilroberta-base'

train_batch_size = 64           #Increasing the train batch size improves the model performance, but requires more GPU memory

num_dev_queries = 500           #Number of queries we want to use to evaluate the performance while training
num_max_dev_negatives = 200     #For every dev query, we use up to 200 hard negatives and add them to the dev corpus



### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)


def read_corpus():
    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')

    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage

    return corpus


def read_queries():
    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')

    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    return queries


def prepare_data_for_evaluation(queries, corpus):
    # We extracted in the train-eval split 500 random queries that can be used for evaluation during training
    train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv')

    dev_queries = {}
    dev_corpus = {}
    dev_rel_docs = {}

    num_negatives = defaultdict(int)

    # with gzip.open(train_eval_filepath, 'rt') as fIn:
    with open(train_eval_filepath, 'r') as fIn:
        for line in fIn:
            qid, pos_id, neg_id = line.strip().split()

            if len(dev_queries) <= num_dev_queries or qid in dev_queries:
                dev_queries[qid] = queries[qid]

                #Ensure the corpus has the positive
                dev_corpus[pos_id] = corpus[pos_id]

                if qid not in dev_rel_docs:
                    dev_rel_docs[qid] = set()

                dev_rel_docs[qid].add(pos_id)

                if num_negatives[qid] < num_max_dev_negatives:
                    dev_corpus[neg_id] = corpus[neg_id]
                    num_negatives[qid] += 1

    logging.info("Dev queries: {}".format(len(dev_queries)))
    logging.info("Dev Corpus: {}".format(len(dev_corpus)))

    return dev_queries, dev_corpus, dev_rel_docs


# We load the qidpidtriples file on-the-fly by using a custom IterableDataset class
class TripletsDataset(IterableDataset):
    def __init__(self, model, queries, corpus, triplets_file):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.triplets_file = triplets_file

    def __iter__(self):
        with open(self.triplets_file, 'r') as fIn:
            for line in fIn:
                qid, pos_id, neg_id = line.strip().split()
                query_text = self.queries[qid]
                pos_text = self.corpus[pos_id]
                neg_text = self.corpus[neg_id]
                yield InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        # return 397226027
        return 10000


def train():
    # We construct the SentenceTransformer bi-encoder from scratch
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model_save_path = 'output/training_ms-marco_bi-encoder-' + model_name.replace(
        "/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Read our training file. qidpidtriples consists of triplets (qid, positive_pid, negative_pid)
    train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv')

    # Create the evaluator that is called during training
    queries = read_queries()
    corpus = read_corpus()
    dev_queries, dev_corpus, dev_rel_docs = prepare_data_for_evaluation(queries, corpus)
    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        dev_queries, dev_corpus, dev_rel_docs, name='ms-marco-train_eval')

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = TripletsDataset(model=model, queries=queries, corpus=corpus, triplets_file=train_filepath)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # print(next(iter(train_dataloader)))
    # return

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=ir_evaluator,
              epochs=1,
              warmup_steps=1000,
              output_path=model_save_path,
              evaluation_steps=5000,
              use_amp=True
              )


# def create_data_loader():
#     code_snippets_file = '../unif/data/parallel_bodies_n1000'
#     descriptions_file = '../unif/data/parallel_desc_n1000'
#     dataset = CodeDescRawTripletDataset(code_snippets_file, descriptions_file)
#
#     # train_dataset = []
#     # for code_snippet, positive_desc, negative_desc in dataset:
#     #     #                                 (query,       positive ans, negative ans)
#     #     inp_example = InputExample(texts=[code_snippet, positive_desc, negative_desc])
#     #     train_dataset.append(inp_example)
#
#     train_data_loader = DataLoader(dataset, shuffle=True, batch_size=train_batch_size)
#
#     return train_data_loader


def prepare_dataset_for_topn_evaluation(dataset: CodeDescRawTripletDataset):

    assert len(dataset.code_snippets) == len(dataset.descriptions),\
        'code_snippets and descriptions must have the same length'

    num_pairs = len(dataset)
    num_digits = len(str(num_pairs))

    code_ids = {'c' + str(i).zfill(num_digits): dataset.code_snippets[i] for i in range(num_pairs)}
    desc_ids = {'d' + str(i).zfill(num_digits): dataset.descriptions[i] for i in range(num_pairs)}

    relevant_docs = {}
    for code_id, query_id in zip(code_ids, desc_ids):
        relevant_docs[query_id] = {code_id}

    return code_ids, desc_ids, relevant_docs


def my_own_train(use_wandb=False):

    model_name = 'msmarco-distilbert-base-v2'
    # train_batch_size = 16
    num_epochs = 8
    model_save_path = 'output/training_ms-marco_bi-encoder-' + model_name.replace(
        "/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(model_name)
    model.max_seq_length = 64
    # 128 or larger does not work to evaluate because it runs out of memory
    model.parallel_tokenization = False
    learning_rate = 5e-5
    warmup_steps = 3000
    evaluation_steps = 2000

    code_snippets_file = '../unif/data/parallel_bodies_n1000'
    descriptions_file = '../unif/data/parallel_desc_n1000'
    dataset = CodeDescRawTripletDataset(code_snippets_file, descriptions_file)
    train_data_loader = DataLoader(dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    code_ids, desc_ids, relevant_docs = prepare_dataset_for_topn_evaluation(dataset)
    # ir_evaluator = InformationRetrievalEvaluator(desc_ids, code_ids, relevant_docs, name='all-train_eval')
    ir_evaluator = WandbInformationRetrievalEvaluator(desc_ids, code_ids, relevant_docs, name='all-train_eval', use_wandb=True)

    if use_wandb:
        wandb.init(project='code-search', name='sbert', reinit=True)
        config = wandb.config
        config.max_seq_length = model.max_seq_length
        config.learning_rate = learning_rate
        config.warmup_steps = warmup_steps
        config.evaluation_steps = evaluation_steps
        # config.embedding_size = embedding_size
        # config.evaluate_size = evaluate_size
        # config.margin = margin
        config.num_epochs = num_epochs
        config.train_size = len(dataset)
        wandb.watch(model)

    model.fit(train_objectives=[(train_data_loader, train_loss)],
              evaluator=ir_evaluator,
              epochs=num_epochs,
              # scheduler = 'warmupconstant',
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              optimizer_params={'lr': learning_rate},
              evaluation_steps=evaluation_steps,
              use_amp=True
              )

    test_evaluator = InformationRetrievalEvaluator(desc_ids, code_ids, relevant_docs, name='all-test_ir')
    test_evaluator(model)


def main():
    # train()
    # data_loader = create_data_loader()
    # iterator = iter(data_loader)
    # print(next(iterator))

    # code_snippets_file = '../unif/data/parallel_bodies_n1000'
    # descriptions_file = '../unif/data/parallel_desc_n1000'
    # dataset = CodeDescRawTripletDataset(code_snippets_file, descriptions_file)
    #
    # print(dataset[0])
    # print(dataset[3])
    # print(dataset[10])

    # print(next(iterator))
    my_own_train(use_wandb=True)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
