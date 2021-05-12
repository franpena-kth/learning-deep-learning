from typing import Dict, Set, List

import wandb
from sentence_transformers.evaluation import InformationRetrievalEvaluator


class WandbInformationRetrievalEvaluator(InformationRetrievalEvaluator):

    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, Set[str]],
                 corpus_chunk_size: int = 50000, mrr_at_k: List[int] = [10], ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10], precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100], show_progress_bar: bool = False, batch_size: int = 32, name: str = '',
                 use_wandb: bool = False):
        super().__init__(queries, corpus, relevant_docs, corpus_chunk_size, mrr_at_k, ndcg_at_k, accuracy_at_k,
                         precision_recall_at_k, map_at_k, show_progress_bar, batch_size, name)
        self.use_wandb = use_wandb

    def output_scores(self, scores):
        super().output_scores(scores)
        if self.use_wandb:
            wandb.log(scores)
