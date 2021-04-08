import time

import keras
import matplotlib
import numpy
from matplotlib import pyplot


def plot_learning_curves(loss, val_loss):
    pyplot.plot(numpy.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    pyplot.plot(numpy.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    pyplot.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    pyplot.axis([1, 20, 0, 0.05])
    pyplot.legend(fontsize=14)
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Loss")
    pyplot.grid(True)


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", n_steps=50):
    pyplot.plot(series, ".-")
    if y is not None:
        pyplot.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        pyplot.plot(n_steps, y_pred, "ro")
    pyplot.grid(True)
    if x_label:
        pyplot.xlabel(x_label, fontsize=16)
    if y_label:
        pyplot.ylabel(y_label, fontsize=16, rotation=0)
    pyplot.hlines(0, 0, 100, linewidth=1)
    pyplot.axis([0, n_steps + 1, -1, 1])


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    pyplot.plot(numpy.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    pyplot.plot(numpy.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    pyplot.axis([0, n_steps + ahead, -1, 1])
    pyplot.legend(fontsize=14)
    pyplot.show()


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = numpy.random.rand(4, batch_size, 1)
    time_var = numpy.linspace(0, 1, n_steps)
    series = 0.5 * numpy.sin((time_var - offsets1) * (freq1 * 10 + 10))   # wave 1
    series += 0.2 * numpy.sin((time_var - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (numpy.random.rand(batch_size, n_steps) - 0.5)    # + noise
    return series[..., numpy.newaxis].astype(numpy.float32)


def create_data_set():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def baseline_last_value():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    y_pred = X_valid[:, -1]
    print(numpy.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


def baseline_linear_regression():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[50, 1]),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20)
    y_pred = model.predict(X_valid)
    print(numpy.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


def simple_rnn():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20)
    y_pred = model.predict(X_valid)
    print(model.summary())
    print(numpy.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


def deep_rnn():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    # model.fit(X_train, y_train, epochs=20)
    # y_pred = model.predict(X_valid)
    # print(numpy.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


def deep_rnn_dense():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    model.fit(X_train, y_train, epochs=20)
    y_pred = model.predict(X_valid)
    print(numpy.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


def deep_rnn_dense_multiple():
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20)

    n_steps = 50
    series = generate_time_series(1, n_steps + 10)
    X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
    X = X_new
    for step_ahead in range(10):
        y_pred_one = model.predict(X[:, step_ahead:])[:, numpy.newaxis, :]
        X = numpy.concatenate([X, y_pred_one], axis=1)

    Y_pred = X[:, n_steps:]

    plot_multiple_forecasts(X, Y_new, Y_pred)


def main():
    # print(generate_time_series(10, 5))
    # baseline_last_value()
    # baseline_linear_regression()
    # simple_rnn()
    # deep_rnn()
    # deep_rnn_dense()
    # deep_rnn_dense_multiple()
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_data_set()
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
