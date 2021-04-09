import re
from collections import Counter

import numpy
import pandas
import seaborn
import torch
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = numpy.random.rand(4, batch_size, 1)
    time = numpy.linspace(0, 1, n_steps)
    series = 0.5 * numpy.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * numpy.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (numpy.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., numpy.newaxis].astype(numpy.float32)


def create_time_series_train_test_sets(sequence_length):
    numpy.random.seed(42)

    # N_STEPS = 50
    series = generate_time_series(10000, sequence_length + 1)
    X_train, y_train = series[:7000, :sequence_length], series[:7000, -1]
    X_test, y_test = series[7000:, :sequence_length], series[7000:, -1]
    X_full, y_full = series[:, :sequence_length], series[:, -1]

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    X_full = torch.from_numpy(X_full)
    y_full = torch.from_numpy(y_full)



    # The dimensionality of the X tensor is (batch_size, sequence_length, num_features)
    # The dimensionality of the y tensor is (batch_size, num_features)

    return X_full, y_full, X_train, y_train, X_test, y_test, None


def load_airline_passengers():
    training_set = pandas.read_csv('airline-passengers.csv')
    #training_set = pd.read_csv('shampoo.csv')

    training_set = training_set.iloc[:,1:2].values

    #plt.plot(training_set, label = 'Shampoo Sales Data')
    pyplot.plot(training_set, label = 'Airline Passangers Data')
    pyplot.show()

    return training_set


def create_airline_passengers_train_test_sets(sequence_length):
    training_set = load_airline_passengers()
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    # seq_length = 4
    x, y = create_sequences(training_data, sequence_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(numpy.array(x)))
    dataY = Variable(torch.Tensor(numpy.array(y)))

    trainX = Variable(torch.Tensor(numpy.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(numpy.array(y[0:train_size])))

    testX = Variable(torch.Tensor(numpy.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(numpy.array(y[train_size:len(y)])))

    return dataX, dataY, trainX, trainY, testX, testY, sc


def create_covid_train_test_sets(sequence_length):
    df = pandas.read_csv('time_series_19-covid-Confirmed.csv')
    df = df.iloc[:, 4:]

    daily_cases = df.sum(axis=0)
    daily_cases.index = pandas.to_datetime(daily_cases.index)
    daily_cases.head()
    pyplot.plot(daily_cases)
    pyplot.title("Cumulative daily cases")
    pyplot.show()

    daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(numpy.int64)
    daily_cases.head()

    pyplot.plot(daily_cases)
    pyplot.title("Daily cases")

    test_data_size = 14
    train_data = daily_cases[:-test_data_size]
    test_data = daily_cases[-test_data_size:]
    full_data = daily_cases.to_numpy()

    scaler = MinMaxScaler()
    scaler = scaler.fit(numpy.expand_dims(train_data, axis=1))
    train_data = scaler.transform(numpy.expand_dims(train_data, axis=1))
    test_data = scaler.transform(numpy.expand_dims(test_data, axis=1))
    full_data = scaler.transform(numpy.expand_dims(full_data, axis=1))

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    X_full, y_full = create_sequences(full_data, sequence_length)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    X_full = torch.from_numpy(X_full).float()
    y_full = torch.from_numpy(y_full).float()

    return X_full, y_full, X_train, y_train, X_test, y_test, scaler


def create_sequences(data, sequence_length):
    xs = []
    ys = []

    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)

    return numpy.array(xs), numpy.array(ys)


def create_flights_train_test_sets(sequence_length):
    flight_data = seaborn.load_dataset("flights")
    flight_data.head()

    fig_size = pyplot.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    pyplot.rcParams["figure.figsize"] = fig_size

    pyplot.title('Month vs Passenger')
    pyplot.ylabel('Total Passengers')
    pyplot.xlabel('Months')
    pyplot.grid(True)
    pyplot.autoscale(axis='x', tight=True)
    pyplot.plot(flight_data['passengers'])

    all_data = flight_data['passengers'].values.astype(float)

    test_data_size = 12

    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    full_data = all_data

    # from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(numpy.expand_dims(train_data, axis=1))
    # train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    # train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    # test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
    # full_data_normalized = torch.FloatTensor(full_data_normalized).view(-1)
    train_data_normalized = scaler.transform(numpy.expand_dims(train_data, axis=1))
    test_data_normalized = scaler.transform(numpy.expand_dims(test_data, axis=1))
    full_data_normalized = scaler.transform(numpy.expand_dims(full_data, axis=1))
    X_train, y_train = create_sequences(train_data_normalized, sequence_length)
    X_test, y_test = create_sequences(test_data_normalized, sequence_length)
    X_full, y_full = create_sequences(full_data_normalized, sequence_length)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    X_full = torch.from_numpy(X_full).float()
    y_full = torch.from_numpy(y_full).float()

    return X_full, y_full, X_train, y_train, X_test, y_test, scaler


def create_sine_train_test_sets(sequence_length):
    data = torch.load('traindata.pt')
    train_input = torch.from_numpy(data[3:, :-1])
    train_target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    full_input = torch.from_numpy(data[:, :-1])
    full_target = torch.from_numpy(data[:, 1:])

    return full_input, full_target, train_input, train_target, test_input, test_target, None


def create_char_dataset():
    data_folder = '/Users/frape/Datasets/uncompressed/literature/'
    data_path = data_folder + "sherlock.txt"
    # data_path = "./shakespeare.txt"
    # data_path = "./parallel_bodies.txt"
    #######################################

    # load the text file
    # data = open(data_path, 'r').read()
    data = open(data_path, 'r').read()[:10000]
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")

    # char to index and index to char maps
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]

    # data tensor on device
    # data = torch.tensor(data).to(device)
    data = torch.tensor(data)
    # Change the dimension of the tensor from (m,) to (m,1)
    data = torch.unsqueeze(data, dim=1)

    return data, char_to_ix, ix_to_char, data_size, vocab_size


def create_word_dataset():
    data_folder = '/Users/frape/Datasets/uncompressed/literature/'
    data_path = data_folder + "sherlock.txt"
    # data_path = "./reddit-cleanjokes.csv"
    # data_path = "./parallel_bodies.txt"
    #######################################

    # load the text file
    # data = open(data_path, 'r').read()
    data = open(data_path, 'r').read()
    data = data.split()[:20000]
    data = ' '.join(data)
    data = data.lower().strip()
    data = re.sub(r"([.!?])", r" \1", data)
    data = re.sub(r"[^a-zA-Z.!?]+", r" ", data)
    from spacy.lang.en import English
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer
    data = [tok.text for tok in tokenizer(data)]

    # print(len(data))
    # print(data[0])
    # print(data[10])
    # print(data[100])
    # print(data)

    words = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(words)
    print("----------------------------------------")
    print("Data has {} words, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")

    # # word to index and index to word maps
    word_to_index = {word: index for index, word in enumerate(words)}
    index_to_word = {index: word for index, word in enumerate(words)}

    # convert data from chars to indices
    data = list(data)
    for index, word in enumerate(data):
        data[index] = word_to_index[word]

    # data tensor on device
    # data = torch.tensor(data).to(device)
    data = torch.tensor(data)
    # Change the dimension of the tensor from (m,) to (m,1)
    data = torch.unsqueeze(data, dim=1)

    return data, word_to_index, index_to_word, data_size, vocab_size


class CharDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_length,
    ):
        self.sequence_length = sequence_length
        self.data = self.load_chars()
        # self.unique_chars = self.get_uniq_chars()

        # self.index_to_char = {index: char for index, char in enumerate(self.unique_chars)}
        # self.char_to_index = {char: index for index, char in enumerate(self.unique_chars)}

        # self.chars_indexes = [self.char_to_index[w] for w in self.chars]

    # def load_chars(self):
    #     train_df = pandas.read_csv('data/reddit-cleanjokes.csv')
    #     text = train_df['Joke'].str.cat(sep=' ')
    #     return text.split(' ')

    def load_chars(self):
        data_folder = '/Users/frape/Datasets/uncompressed/literature/'
        data_path = data_folder + "sherlock.txt"
        # data_path = "./parallel_bodies.txt"
        data = open(data_path, 'r').read()[:10000]

        self.unique_chars = sorted(list(set(data)))
        self.index_to_char = {index: char for index, char in enumerate(self.unique_chars)}
        self.char_to_index = {char: index for index, char in enumerate(self.unique_chars)}
        # self.chars_indexes = [self.char_to_index[w] for w in chars]

        data = list(data)
        for i, ch in enumerate(data):
            data[i] = self.char_to_index[ch]
        data = torch.tensor(data)
        # Change the dimension of the tensor from (m,) to (m,1)
        data = torch.unsqueeze(data, dim=1)

        return data

    # def get_uniq_chars(self):
    #     char_counts = Counter(self.chars)
    #     return sorted(char_counts, key=char_counts.get, reverse=True)

    def __len__(self):
        # return len(self.chars_indexes) - self.sequence_length
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        # Returns the char at position index and the following char
        # For a sequence model this is (x, y)
        # return (
        #     torch.tensor(self.chars_indexes[index:index + self.sequence_length]),
        #     torch.tensor(self.chars_indexes[index + 1:index + self.sequence_length + 1]),
        # )
        # return (
        #     torch.tensor(self.data[index:index + self.sequence_length]),
        #     torch.tensor(self.data[index + 1:index + self.sequence_length + 1]),
        # )
        # print('data', self.data[index:index + self.sequence_length].shape)
        return (
            self.data[index:index + self.sequence_length],
            self.data[index + 1:index + self.sequence_length + 1],
        )

# create_word_dataset()
