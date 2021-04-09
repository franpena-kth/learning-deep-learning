import torch
from matplotlib import pyplot as plt

from nlp_from_scratch.generating_names.gn_data import DataProcessor
from nlp_from_scratch.generating_names.gn_train import train_cycle


def plot(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()


# Sample from a category and starting letter
def sample(model, data_processor, category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = data_processor.categoryTensor(category)
        input = data_processor.inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        max_length = 20

        for i in range(max_length):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == data_processor.n_letters - 1:
                break
            else:
                letter = data_processor.all_letters[topi]
                output_name += letter
            input = data_processor.inputTensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(model, data_processor, category, start_letters='ABC'):
    print(f'\n{category}')
    for start_letter in start_letters:
        print(sample(model, data_processor, category, start_letter))


def main():
    # plant_seeds()
    data_processor = DataProcessor()
    model, current_loss, all_losses = train_cycle(data_processor)
    plot(all_losses)
    samples(model, data_processor, 'Russian', 'RUS')
    samples(model, data_processor, 'German', 'GER')
    samples(model, data_processor, 'Spanish', 'SPA')
    samples(model, data_processor, 'Chinese', 'CHI')


main()
