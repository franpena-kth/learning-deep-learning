import random

import numpy
import torch
import matplotlib.pyplot as plt


############################
### PLOTTING THE RESULTS ###
############################
from matplotlib import ticker

from nlp_from_scratch.names_classifier.nc_data import DataProcessor
from nlp_from_scratch.names_classifier.nc_model import RNN
from nlp_from_scratch.names_classifier.nc_train import train_cycle


def plot(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()


##############################
### EVALUATING THE RESULTS ###
##############################




# Just return an output given a line
def evaluate(model, line_tensor):
    model.initHidden()
    output = model(line_tensor)

    return output


def plot_confusion_matrix(model):
    data_processor = DataProcessor()

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(data_processor.n_categories, data_processor.n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = data_processor.randomTrainingExample()
        output = evaluate(model, line_tensor)
        guess, guess_i = data_processor.categoryFromOutput(output)
        category_i = data_processor.all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(data_processor.n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + data_processor.all_categories, rotation=90)
    ax.set_yticklabels([''] + data_processor.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


##################################
### TURNING NAMES INTO TENSORS ###
##################################

def predict(model, input_line, n_predictions=3):
    data_processor = DataProcessor()
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(model, data_processor.lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, data_processor.all_categories[category_index]))
            predictions.append([value, data_processor.all_categories[category_index]])


def plant_seeds():
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)


#############################
### RUNNING ON USER INPUT ###
#############################

def main():
    plant_seeds()
    data_processor = DataProcessor()
    model, current_loss, all_losses = train_cycle(data_processor)
    plot(all_losses)
    predict(model, 'Dovesky')
    predict(model, 'Jackson')
    predict(model, 'Satoshi')


main()
