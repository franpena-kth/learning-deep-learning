import time

import numpy
from matplotlib import pyplot

import data_loader
import utils


def initialize_parameters(n_dims):

    # The shape of W is (n_l, n_l-1). In the logistic regression case, the output layer size is n_1=1 and the input
    # layer size is n_0 = X.shape[0]. There 
    w = numpy.zeros((n_dims, 1))
    b = 0

    assert (w.shape == (n_dims, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def calculate_Z(X, w, b):

    return numpy.dot(w.T, X) + b


def calculate_A(Z):

    return utils.sigmoid(Z)


def forward_step(X, w, b):

    A = calculate_A(calculate_Z(X, w, b))
    return A


def calculate_cost(A, Y):

    m = Y.shape[1]
    cost = -(1.0/m) * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1 - A))

    return cost


def backward_step(A, X, Y,):
    m = X.shape[1]
    dw = (1.0/m) * numpy.dot(X, (A-Y).T)
    db = (1.0/m) * numpy.sum(A-Y)

    grads = {"dw": dw, "db": db}

    return grads


def propagate(w, b, X, Y):

    z = calculate_Z(X, w, b)
    A = calculate_A(z)

    cost = calculate_cost(A, Y)

    grads = backward_step(A, X, Y)

    assert (grads["dw"].shape == w.shape)
    assert (grads["db"].dtype == float)
    cost = numpy.squeeze(cost)
    assert (cost.shape == ())

    return grads, cost


def update_parameters(W, b, dW, db, learning_rate):

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    print(costs)
    print(len(costs))

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = numpy.zeros((1, m))
    A = calculate_A(calculate_Z(X, w, b))

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_parameters(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - numpy.mean(numpy.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - numpy.mean(numpy.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    # print(costs)
    # print(len(costs))

    return d


def plot_costs(d):
    # Plot learning curve (with costs)
    costs = numpy.squeeze(d['costs'])
    pyplot.plot(costs)
    pyplot.ylabel('cost')
    pyplot.xlabel('iterations (per hundreds)')
    pyplot.title("Learning rate =" + str(d["learning_rate"]))
    pyplot.show()


def search_learning_rates():
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data_loader.load_dataset()
    train_set_x, test_set_x = data_loader.preprocess_dataset(train_set_x_orig, test_set_x_orig)

    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        pyplot.plot(numpy.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    pyplot.ylabel('cost')
    pyplot.xlabel('iterations (hundreds)')

    legend = pyplot.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    pyplot.show()


def main():
    # train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # train_set_x, test_set_x = preprocess_dataset(train_set_x_orig, test_set_x_orig)
    #
    # d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
    #           print_cost=False)
    # plot_costs(d)
    search_learning_rates()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
