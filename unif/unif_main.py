import time

import utils
from unif.unif_evaluate import evaluate_top_n
from unif.unif_plots import plot
from unif.unif_random_model import RandomModel
from unif.unif_train import train_cycle


def test_random_model():
    random_model = RandomModel(embedding_size=128)
    evaluate_top_n(random_model)


def test_unif_model():
    unif_model, current_loss, all_losses = train_cycle()
    plot(all_losses)
    evaluate_top_n(unif_model)


def main():
    utils.plant_seeds()
    test_unif_model()
    # test_random_model()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
