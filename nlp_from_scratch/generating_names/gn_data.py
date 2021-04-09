import glob
import os
import random
import string
import unicodedata

import torch


def findFiles(path): return glob.glob(path)


class DataProcessor:

    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters) + 1  # Plus EOS marker
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
        # Build the category_lines dictionary, a list of lines per category
        data_folder = '/Users/frape/Datasets/uncompressed/pytorch-nlp-from-scratch/'
        for filename in findFiles(data_folder + 'names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)

        if self.n_categories == 0:
            raise RuntimeError('Data not found. Make sure that you downloaded data '
                'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                'the current directory.')

        print('# categories:', self.n_categories, self.all_categories)
        print(self.unicodeToAscii("O'Néàl"))

    # Random item from a list
    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    # Get a random category and random line from that category
    def randomTrainingPair(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        return category, line

    # One-hot vector for category
    def categoryTensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1)  # EOS
        return torch.LongTensor(letter_indexes)

    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor
