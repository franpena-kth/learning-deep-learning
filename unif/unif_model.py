import math

import torch
from torch import nn


class UNIF(nn.Module):

    def __init__(self, code_vocab_size, desc_vocab_size, embedding_size):
        super(UNIF, self).__init__()
        self.code_embedding_layer = nn.Embedding(num_embeddings=code_vocab_size, embedding_dim=embedding_size)
        self.desc_embedding_layer = nn.Embedding(num_embeddings=desc_vocab_size, embedding_dim=embedding_size)
        # attention_weights = torch.nn.init.uniform_(
        #     torch.empty(embedding_size, 1, dtype=torch.float32, requires_grad=True))
        # self.attention_weights = nn.parameter.Parameter(attention_weights, requires_grad=True)
        self.attention_weights = nn.Parameter(torch.Tensor(embedding_size, 1))
        bound = 1 / math.sqrt(embedding_size)
        nn.init.uniform_(self.attention_weights, -bound, bound)
        self.softmax = nn.Softmax(dim=-1)
        # self.minus_inf = torch.tensor([[-float('inf')]], device=get_best_device())  # 1 x 1
        self.minus_inf = torch.tensor([[-float('inf')]])  # 1 x 1

    def forward(self, code_token_ids, code_mask, desc_token_ids, desc_mask):

        code_embedding = self.code_embedding_layer(code_token_ids)
        desc_embedding = self.desc_embedding_layer(desc_token_ids)

        # Calculate the attention weights for the code embedding layer
        batch_size = code_token_ids.shape[0]
        attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        attention_scores = torch.where(code_mask.view(batch_size, -1) != 0., torch.bmm(code_embedding, attention_weights).squeeze(-1), self.minus_inf)
        attention_weights = self.softmax(attention_scores.squeeze(-1))
        code_embedding = (attention_weights.unsqueeze(-1) * code_embedding).sum(1)

        # Calculate the average of the embeddings for the desc embedding layer
        desc_embedding *= desc_mask.unsqueeze(-1)
        desc_embedding = desc_embedding.sum(axis=1) / desc_mask.sum(axis=1).unsqueeze(-1)

        return code_embedding, desc_embedding
