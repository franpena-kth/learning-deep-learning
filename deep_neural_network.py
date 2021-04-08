import time

import numpy
from matplotlib import pyplot

import dnn_utils_v2
import dnn_app_utils_v3
import test_cases_v4a


def initialize_parameters_deep(layer_dims, seed=3):

    numpy.random.seed(seed)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # parameters["W" + str(l)] = numpy.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["W" + str(l)] = numpy.random.randn(layer_dims[l], layer_dims[l-1]) / numpy.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = numpy.zeros((layer_dims[l], 1))

    assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):

    Z = numpy.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = dnn_utils_v2.sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = dnn_utils_v2.relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A, W, b, 'relu')
        caches.append(cache)

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -(1/m) * numpy.sum(Y * numpy.log(AL) + (1-Y) * numpy.log(1-AL))

    cost = numpy.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * numpy.dot(dZ, A_prev.T)
    db = (1/m) * numpy.sum(dZ, axis=1, keepdims=True)
    dA_prev = numpy.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = dnn_utils_v2.sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = dnn_utils_v2.relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = - (numpy.divide(Y, AL) - numpy.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

    dA = dAL
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] =\
        linear_activation_backward(dA, caches[L-1], "sigmoid")

    for l in reversed(range(L-1)):
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] =\
            linear_activation_backward(grads["dA" + str(l+1)], caches[l], "relu")

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    numpy.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = dnn_app_utils_v3.initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)
        dA2 = - (numpy.divide(Y, A2) - numpy.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, numpy.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        # plot the cost

    pyplot.plot(numpy.squeeze(costs))
    pyplot.ylabel('cost')
    pyplot.xlabel('iterations (per hundreds)')
    pyplot.title("Learning rate =" + str(learning_rate))
    pyplot.show()

    return parameters


# def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
#     numpy.random.seed(1)
#     costs = []  # keep track of cost
#     parameters = initialize_parameters_deep(layers_dims)
#
#     for i in range(num_iterations):
#         AL, caches = L_model_forward(X, parameters)
#         cost = compute_cost(AL, Y)
#         grads = L_model_backward(AL, Y, caches)
#         update_parameters(parameters, grads, learning_rate)
#
#         # Print the cost every 100 training example
#         if print_cost and i % 100 == 0:
#             print("Cost after iteration %i: %f" % (i, cost))
#         if print_cost and i % 100 == 0:
#             costs.append(cost)
#
#     # plot the cost
#     pyplot.plot(numpy.squeeze(costs))
#     pyplot.ylabel('cost')
#     pyplot.xlabel('iterations (per hundreds)')
#     pyplot.title("Learning rate =" + str(learning_rate))
#     pyplot.show()
#
#     return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    numpy.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims, 1)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    pyplot.plot(numpy.squeeze(costs))
    pyplot.ylabel('cost')
    pyplot.xlabel('iterations (per hundreds)')
    pyplot.title("Learning rate =" + str(learning_rate))
    pyplot.show()

    return parameters


def prediction(parameters, X):

    pass
    # return prediction


def optimize():

    pass


def nn_model():

    pass


def test_case_initialize_paramters():
    parameters = initialize_parameters_deep([5, 4, 3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def test_case_linear_forward():
    A, W, b = test_cases_v4a.linear_forward_test_case()

    Z, linear_cache = linear_forward(A, W, b)
    print("Z = " + str(Z))


def test_case_linear_activation_forward():
    A_prev, W, b = test_cases_v4a.linear_activation_forward_test_case()

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    print("With sigmoid: A = " + str(A))

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    print("With ReLU: A = " + str(A))


def test_case_L_model_forward():
    X, parameters = test_cases_v4a.L_model_forward_test_case_2hidden()
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))


def test_case_compute_cost():
    Y, AL = test_cases_v4a.compute_cost_test_case()
    print("cost = " + str(compute_cost(AL, Y)))


def test_case_linear_backward():
    # Set up some test inputs
    dZ, linear_cache = test_cases_v4a.linear_backward_test_case()

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db))


def test_case_linear_activation_backward():
    dAL, linear_activation_cache = test_cases_v4a.linear_activation_backward_test_case()

    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
    print("sigmoid:")
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db) + "\n")

    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
    print("relu:")
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db))


def test_case_L_model_backward():
    AL, Y_assess, caches = test_cases_v4a.L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    test_cases_v4a.print_grads(grads)


def test_case_update_paramters():
    parameters, grads = test_cases_v4a.update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def test_case_two_layer_model():
    train_x_orig, train_y, test_x_orig, test_y, classes = dnn_app_utils_v3.load_data()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True)


def test_case_L_layer_model():
    train_x_orig, train_y, test_x_orig, test_y, classes = dnn_app_utils_v3.load_data()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    ### CONSTANTS ###
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    predictions_train = dnn_app_utils_v3.predict(train_x, train_y, parameters)
    predictions_test = dnn_app_utils_v3.predict(test_x, test_y, parameters)


def main():
    # test_case_initialize_paramters()
    # test_case_linear_forward()
    # test_case_linear_activation_forward()
    # test_case_L_model_forward()
    # test_case_compute_cost()
    # test_case_linear_backward()
    # test_case_linear_activation_backward()
    # test_case_L_model_backward()
    # test_case_update_paramters()
    # test_case_two_layer_model()
    test_case_L_layer_model()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
