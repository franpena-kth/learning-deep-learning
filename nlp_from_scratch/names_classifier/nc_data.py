import glob
import os
import random
import string
import unicodedata

import torch

from torch.utils.data import Dataset


def findFiles(path):
    return glob.glob(path)

# print(findFiles('data/names/*.txt'))


class DataProcessor:

    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        # Build the category_lines dictionary, a list of names per language
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = 0
        self.init_categories()

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def init_categories(self):
        data_folder = '/Users/frape/Datasets/uncompressed/pytorch-nlp-from-scratch/'
        for filename in findFiles(data_folder + 'names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)
        # print(self.category_lines['Italian'][:5])

    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor




##################################
### TURNING NAMES INTO TENSORS ###
##################################


# data_processor = DataProcessor()
# category, line, category_tensor, line_tensor = data_processor.randomTrainingExample()
# print(category)
# print(line)
# print(category_tensor)
# print(line_tensor)
#
# for i in range(10):
#     category, line, category_tensor, line_tensor = data_processor.randomTrainingExample()
#     print('category =', category, '/ line =', line)
#
#
# print(data_processor.category_lines['Italian'][:5])
# print(data_processor.letterToTensor('J'))
# print(data_processor.lineToTensor('Jones').size())
# print(data_processor.categoryFromOutput(output))




class NamesDataset(Dataset):

    def __init__(self):
        self.data_processor = DataProcessor()

    def __getitem__(self, item):
        return self.data_processor.randomTrainingExample()

    def __len__(self):
        return 100000


