
import math

import torch
from torch import nn


class RandomModel(nn.Module):

    def __init__(self, embedding_size):
        super(RandomModel, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, code_token_ids, code_mask, desc_token_ids, desc_mask):

        batch_size = code_token_ids.shape[0]
        code_embedding = torch.rand(batch_size, self.embedding_size)
        desc_embedding = torch.rand(batch_size, self.embedding_size)

        return code_embedding, desc_embedding
