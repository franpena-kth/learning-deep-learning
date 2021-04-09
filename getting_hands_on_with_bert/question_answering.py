
from transformers import BertForQuestionAnswering, BertTokenizer
import torch


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


question = "What is the immune system?"
paragraph = "The immune system is a system of many biological structures and " \
            "processes within an organism that protects against disease. " \
            "To function properly, an immune system must detect a wide " \
            "variety of agents, known as pathogens, from viruses to " \
            "parasitic worms, and distinguish them from the organism's own " \
            "healthy tissue."

question = '[CLS] ' + question + ' [SEP]'
paragraph = paragraph + '[SEP]'

question_tokens = tokenizer.tokenize(question)
paragraph_tokens = tokenizer.tokenize(paragraph)

tokens = question_tokens + paragraph_tokens
input_ids = tokenizer.convert_tokens_to_ids(tokens)

segment_ids = [0] * len(question_tokens)
segment_ids += [1] * len(paragraph_tokens)

input_ids = torch.tensor([input_ids])
segment_ids = torch.tensor([segment_ids])

start_scores, end_scores = model(input_ids, token_type_ids=segment_ids).values( )
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
print(' '.join(tokens[start_index:end_index+1]))
