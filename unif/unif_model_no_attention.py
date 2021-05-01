
from torch import nn


class UNIFNoAttention(nn.Module):

    def __init__(self, code_vocab_size, desc_vocab_size, embedding_size):
        super(UNIFNoAttention, self).__init__()
        self.code_embedding_layer = nn.Embedding(num_embeddings=code_vocab_size, embedding_dim=embedding_size)
        self.desc_embedding_layer = nn.Embedding(num_embeddings=desc_vocab_size, embedding_dim=embedding_size)

    def forward(self, code_token_ids, code_mask, desc_token_ids, desc_mask):

        code_embedding = self.code_embedding_layer(code_token_ids)
        desc_embedding = self.desc_embedding_layer(desc_token_ids)

        # Calculate the attention weights for the code embedding layer
        # Calculate the average of the embeddings for the code embedding layer
        code_embedding *= code_mask.unsqueeze(-1)
        code_embedding = code_embedding.sum(axis=1) / code_mask.sum(axis=1).unsqueeze(-1)

        # Calculate the average of the embeddings for the desc embedding layer
        desc_embedding *= desc_mask.unsqueeze(-1)
        desc_embedding = desc_embedding.sum(axis=1) / desc_mask.sum(axis=1).unsqueeze(-1)

        return code_embedding, desc_embedding
