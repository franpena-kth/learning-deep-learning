
from transformers import BertForSequenceClassification, BertTokenizerFast,\
    Trainer, TrainingArguments
from nlp import load_dataset
import torch
import numpy

dataset = load_dataset('csv', data_files='./imdbs.csv', split='train')
print(type(dataset))

dataset = dataset.train_test_split(test_size=0.3)
print(dataset)

train_set = dataset['train']
test_set = dataset['test']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

sentence = 'I love Paris'
tokens = tokenizer([sentence, 'birds fly'], padding=True, max_length=5)
# tokens = ['[CLS]'] + tokens + ['[SEP]']

token_type_ids = [0] * len(tokens)
attention_mask = [1] * len(tokens)

print(tokens)


def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)


train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

print(train_set)
print(type(train_set))

train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

batch_size = 8
epochs = 2

warmup_steps = 500
warmup_decay = 0.01

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=warmup_decay,
    # evaluate_during_training=True,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set
)

trainer.train()

# print(trainer.evaluate())
trainer.evaluate()

