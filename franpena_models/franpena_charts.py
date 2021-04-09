import numpy
from matplotlib import pyplot


def plot_series(sequence_length, series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    pyplot.plot(series, ".-")
    if y is not None:
        pyplot.plot(sequence_length, y, "bx", markersize=10)
    if y_pred is not None:
        pyplot.plot(sequence_length, y_pred, "ro")
    pyplot.grid(True)
    if x_label:
        pyplot.xlabel(x_label, fontsize=16)
    if y_label:
        pyplot.ylabel(y_label, fontsize=16, rotation=0)
    pyplot.hlines(0, 0, 100, linewidth=1)
    pyplot.axis([0, sequence_length + 1, -1, 1])
    pyplot.show()


def make_plots(X_test, y_test):
    fig, axes = pyplot.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    for col in range(3):
        pyplot.sca(axes[col])
        plot_series(X_test[col, :, 0], y_test[col, 0], y_label=("$x(t)$" if col==0 else None))
    pyplot.show()


def plot_true_vs_predicted(dataY, predicted_test, train_size, scaler=None, max_points=200):
    data_predict = predicted_test.data.numpy()
    dataY_plot = dataY.data.numpy()

    if scaler is not None:
        data_predict = scaler.inverse_transform(data_predict)
        dataY_plot = scaler.inverse_transform(dataY_plot)

    # train_size = 100
    if train_size > max_points:
        train_size = max_points
    pyplot.axvline(x=train_size, c='r', linestyle='--')

    if dataY.size(0) > max_points:
        print(f'Warning: showing the first {max_points} data points out of {dataY.size(0)}')

    pyplot.plot(dataY_plot[:max_points])
    pyplot.plot(data_predict[:max_points])
    pyplot.suptitle('Time-Series Prediction')
    pyplot.show()
