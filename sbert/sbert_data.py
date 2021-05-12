import random

from sentence_transformers import InputExample
from torch.utils.data import Dataset


class CodeDescRawTripletDataset(Dataset):

    def __init__(self, code_snippets_file, descriptions_file, size=None):

        # Load data here
        with open(code_snippets_file) as f:
            if size is None:
                self.code_snippets = [line.rstrip() for line in f]
            else:
                self.code_snippets = [line.rstrip() for line in f][:size]

        with open(descriptions_file, encoding="ISO-8859-1") as f:
            if size is None:
                self.descriptions = [line.rstrip() for line in f]
            else:
                self.descriptions = [line.rstrip() for line in f][:size]

        assert len(self.code_snippets) == len(
            self.descriptions), 'The code snippets file must have the same size as the descriptions file'

    def __len__(self):
        # Return dataset size
        return len(self.code_snippets)

    def __getitem__(self, index):
        # Load the code and the descriptions
        code_snippet = self.code_snippets[index]
        positive_desc = self.descriptions[index]
        # negative_desc = self.descriptions[random.randint(0, self.__len__() - 1)]
        negative_candidates = list(range(self.__len__()))
        negative_candidates.remove(index)
        negative_index = random.choice(negative_candidates)
        negative_desc = self.descriptions[negative_index]

        input_example = InputExample(texts=[code_snippet, positive_desc, negative_desc])

        return input_example
