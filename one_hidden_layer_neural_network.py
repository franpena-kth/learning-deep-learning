import time

import numpy

import data_loader
from matplotlib import pyplot

import test_cases
import utils


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):

    numpy.random.seed(2)

    W1 = numpy.random.randn(n_h, n_x) * 0.01
    b1 = numpy.zeros((n_h, 1))
    W2 = numpy.random.randn(n_y, n_h) * 0.01
    b2 = numpy.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = numpy.dot(W1, X) + b1
    A1 = numpy.tanh(Z1)
    Z2 = numpy.dot(W2, A1) + b2
    A2 = utils.sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):

    m = Y.shape[1]
    cost = -(1.0/m) * numpy.sum(Y * numpy.log(A2) + (1 - Y) * numpy.log(1 - A2))

    cost = float(numpy.squeeze(cost))  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    assert(isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    pass

    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * numpy.dot(dZ2, A1.T)
    db2 = (1/m) * numpy.sum(dZ2, axis=1, keepdims=True)
    dZ1 = numpy.multiply(numpy.dot(W2.T, dZ2), (1 - numpy.power(A1, 2)))
    dW1 = (1/m) * numpy.dot(dZ1, X.T)
    db1 = (1/m) * numpy.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):

    numpy.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)
    # print(parameters)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        # print(grads)
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    # print(parameters)

    return parameters


def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions


def visualize_data():
    X, Y = data_loader.load_planar_dataset()
    Y_vector = Y.reshape(Y.shape[1])

    # Visualize the data:
    pyplot.scatter(X[0, :], X[1, :], c=Y_vector, s=40, cmap=pyplot.cm.Spectral)
    pyplot.show()


def build_nn_model():
    # Build a model with a n_h-dimensional hidden layer
    X, Y = data_loader.load_planar_dataset()
    Y_vector = Y.reshape(Y.shape[1])
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

    # Plot the decision boundary
    data_loader.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y_vector)
    pyplot.title("Decision Boundary for hidden layer size " + str(4))
    pyplot.show()


def tuning_nn_model():
    # This may take about 2 minutes to run
    X, Y = data_loader.load_planar_dataset()
    Y_vector = Y.reshape(Y.shape[1])

    pyplot.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        pyplot.subplot(5, 2, i + 1)
        pyplot.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        data_loader.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y_vector)
        predictions = predict(parameters, X)
        accuracy = float((numpy.dot(Y, predictions.T) + numpy.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    pyplot.show()


def other_datasets():
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = data_loader.load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    ### START CODE HERE ### (choose your dataset)
    # dataset = "noisy_circles"
    # dataset = "noisy_moons"
    # dataset = "blobs"
    dataset = "gaussian_quantiles"
    ### END CODE HERE ###

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    Y_vector = Y.reshape(Y.shape[1])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    # Visualize the data
    pyplot.scatter(X[0, :], X[1, :], c=Y_vector, s=40, cmap=pyplot.cm.Spectral)
    pyplot.show()



def test_case_forward_propagation():
    # Test case forward propagation
    X_assess, parameters = test_cases.forward_propagation_test_case()
    A2, cache = forward_propagation(X_assess, parameters)

    # Note: we use the mean here just to make sure that your output matches ours.
    print(numpy.mean(cache['Z1']), numpy.mean(cache['A1']), numpy.mean(cache['Z2']), numpy.mean(cache['A2']))


def test_case_compute_cost():
    # Test case compute cost
    A2, Y_assess, parameters = test_cases.compute_cost_test_case()
    print("cost = " + str(compute_cost(A2, Y_assess)))


def test_case_backward_propagation():
    # Test case backward propagation
    parameters, cache, X_assess, Y_assess = test_cases.backward_propagation_test_case()

    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))


def test_case_update_parameters():
    # Test case update parameters
    parameters, grads = test_cases.update_parameters_test_case()
    parameters = update_parameters(parameters, grads)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def test_case_nn_model():
    # Test nn model
    X_assess, Y_assess = test_cases.nn_model_test_case()
    parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def test_case_predictions():
    parameters, X_assess = test_cases.predict_test_case()

    predictions = predict(parameters, X_assess)
    print("predictions mean = " + str(numpy.mean(predictions)))


def main():
    # numpy.random.seed(1)  # set a seed so that the results are consistent
    # visualize_data()

    # test_case_forward_propagation()
    # test_case_compute_cost()
    # test_case_backward_propagation()
    # test_case_compute_cost()
    # test_case_nn_model()
    # test_case_predictions()

    # build_nn_model()
    # tuning_nn_model()
    other_datasets()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
