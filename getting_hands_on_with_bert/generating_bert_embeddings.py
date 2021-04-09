
from transformers import BertModel, BertTokenizer
import torch


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = 'I love Paris'
tokens = tokenizer.tokenize(sentence)

print(tokens)

tokens = ['[CLS]'] + tokens + ['[SEP]']

print(tokens)

# We add pads so that list of tokens has a length of 7
tokens = tokens + ['[PAD]', '[PAD]']

print(tokens)

attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]

print(attention_mask)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(token_ids)

token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

hidden_rep, cls_head = model(token_ids, attention_mask=attention_mask).values()

# print(hidden_rep)
# print(cls_head)
print(hidden_rep.shape)
print(cls_head.shape)
