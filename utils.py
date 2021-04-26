import random

import numpy
import torch


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + numpy.exp(-x))
    return s


def plant_seeds():
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)


def get_best_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

