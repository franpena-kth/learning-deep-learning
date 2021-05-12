import time

import utils
from unif.unif_evaluate import evaluate_top_n
from unif.unif_plots import plot
from unif.unif_random_model import RandomModel
from unif import unif_train_triplet, unif_train_cosine_neg, unif_train_cosine_pos_neg
from unif import unif_train_cosine_pos


def test_random_model():
    random_model = RandomModel(embedding_size=128)
    evaluate_top_n(random_model)


def test_unif_model():
    unif_model, current_loss, all_losses = unif_train_triplet.train_cycle()
    plot(all_losses)
    evaluate_top_n(unif_model)


def test_unif_cosine_pos_model():
    unif_model, current_loss, all_losses = unif_train_cosine_pos.train_cycle(True)
    plot(all_losses)
    evaluate_top_n(unif_model)


def test_unif_cosine_neg_model():
    unif_model, current_loss, all_losses = unif_train_cosine_neg.train_cycle(True)
    # plot(all_losses)
    # evaluate_top_n(unif_model)


def test_unif_cosine_pos_neg_model():
    unif_model, current_loss, all_losses = unif_train_cosine_pos_neg.train_cycle(True)
    # plot(all_losses)
    # evaluate_top_n(unif_model)


def test_unif_cosine_pos_neg_random_model():
    unif_model, current_loss, all_losses = unif_train_cosine_pos_neg.train_cycle(True)
    # plot(all_losses)
    # evaluate_top_n(unif_model)


def test_unif_triplet_model():
    unif_model, current_loss, all_losses = unif_train_triplet.train_cycle(True)
    # plot(all_losses)
    # evaluate_top_n(unif_model)


def main():
    utils.plant_seeds()
    # test_unif_model()
    # test_unif_cosine_pos_model()
    # test_unif_cosine_neg_model()
    # test_unif_cosine_pos_neg_model()
    test_unif_triplet_model()
    # test_random_model()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
