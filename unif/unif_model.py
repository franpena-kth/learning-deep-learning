
import torch
from torch import nn


class UNIF(nn.Module):

    def __init__(self, code_vocab_size, desc_vocab_size, embedding_size):
        super(UNIF, self).__init__()
        self.code_embedding_layer = nn.Embedding(num_embeddings=code_vocab_size, embedding_dim=embedding_size)
        self.desc_embedding_layer = nn.Embedding(num_embeddings=desc_vocab_size, embedding_dim=embedding_size)
        attention_weights = torch.nn.init.uniform_(
            torch.empty(embedding_size, 1, dtype=torch.float32, requires_grad=True))
        self.attention_weights = nn.parameter.Parameter(attention_weights, requires_grad=True)
        # self.cosine_layer = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, code_token_ids, desc_token_ids):
        # print('Max code', torch.max(code_token_ids))
        # print('Max desc', torch.max(desc_token_ids))

        # print('Code embedding layer', self.code_embedding_layer, self.code_embedding_layer.num_embeddings, self.code_embedding_layer.embedding_dim)

        # print('Code tokens IDs', code_token_ids.shape)
        # print('Desc tokens IDs', desc_token_ids.shape)

        code_embedding = self.code_embedding_layer(code_token_ids)
        desc_embedding = self.desc_embedding_layer(desc_token_ids)

        # Calculate the attention weights for the code embedding layer
        # code_embedding = self.attention_weights(code_embedding)
        # print('Attention weights pre', self.attention_weights.shape)
        # print('Code embedding', code_embedding.shape)
        batch_size = code_token_ids.shape[0]
        attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        # attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        # print('Attention weights post', attention_weights.shape)
        # code_embedding = functional.softmax(torch.bmm(attention_weights, code_embedding))
        # code_embedding = torch.bmm(code_embedding, attention_weights)
        # attention_scores = torch.where(mask.view(batch_size, -1) != 0., torch.bmm(context, query).squeeze(-1), self.minus_inf)
        attention_scores = torch.bmm(code_embedding, attention_weights).squeeze(-1)
        # print('Attention scores', attention_scores.shape)
        # code_embedding = torch.bmm(attention_weights, code_embedding)
        attention_weights = self.softmax(attention_scores.squeeze(-1))
        # print('Attention weights softmax', attention_weights.shape)

        # code_embedding = torch.sum(code_embedding, 1)
        code_embedding = (attention_weights.unsqueeze(-1) * code_embedding).sum(1)
        # print('Code embedding sum', code_embedding.shape)

        # Calculate the average of the embeddings for the desc embedding layer
        desc_embedding = torch.mean(desc_embedding, 1)

        # output = self.cosine_layer(code_embedding, desc_embedding)

        # print('Code embedding', code_embedding.shape)
        # print('Desc embedding', desc_embedding.shape)

        return code_embedding, desc_embedding

    def encode_code_snippet(self, code_token_ids):
        code_embedding = self.code_embedding_layer(code_token_ids)
        batch_size = 1
        attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        code_embedding = torch.bmm(code_embedding, attention_weights)

        return code_embedding

    def encode_description(self, desc_token_ids):
        desc_embedding = self.desc_embedding_layer(desc_token_ids)
        return desc_embedding
