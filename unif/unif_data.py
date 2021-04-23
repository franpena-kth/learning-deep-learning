
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from unif.unif_tokenizer import CuBertHugTokenizer

# MODEL_VOCAB = './vocabulary/python_vocabulary.txt'
MODEL_VOCAB = '../vocabulary/python_vocabulary.txt'
MAX_SEQUENCE_LENGTH = 512
DATASET_SIZE = 10000


class CodeDescDataset(Dataset):

    def __init__(self, code_snippets_file, descriptions_file, size=None):
        self.code_tokenizer = CuBertHugTokenizer(MODEL_VOCAB)
        self.desc_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.code_vocab_size = self.code_tokenizer.vocab_size
        self.desc_vocab_size = self.desc_tokenizer.vocab_size

        if size is None:
            size = DATASET_SIZE

        # Load data here
        # code_snippets_file = './data/parallel_bodies'
        with open(code_snippets_file) as f:
            self.code_snippets = [line.rstrip() for line in f][:size]

        # descriptions_file = './data/parallel_desc'
        with open(descriptions_file, encoding="ISO-8859-1") as f:
            self.descriptions = [line.rstrip() for line in f][:size]

        assert len(self.code_snippets) == len(
            self.descriptions), 'The code snippets file must have the same size as the descriptions file'

    def __len__(self):
        # Return dataset size
        return len(self.code_snippets)

    def __getitem__(self, index):
        # Load the code and the descriptions
        code_snippet = self.code_snippets[index]
        description = self.descriptions[index]

        # Return a (code_token_ids, desc_token_ids) tuple
        tokenized_code = self.code_tokenizer(code_snippet)
        tokenized_desc = self.desc_tokenizer(
            description, padding='max_length', add_special_tokens=True,
            max_length=MAX_SEQUENCE_LENGTH, return_tensors='pt',
            truncation=True)

        return tokenized_code, tokenized_desc