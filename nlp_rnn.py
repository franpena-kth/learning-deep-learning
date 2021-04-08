import time

import keras
import numpy
import tensorflow


def load_char_level_dataset():
    text_url = "https://homl.info/shakespeare"  # shortcut URL
    # text_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    filepath = '/Users/fpena/Datasets/code-docstring-corpus/parallel_bodies'
    with open(filepath) as f:
        text = f.read()
        text = text[:10000]
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)

    print(tokenizer.texts_to_sequences(["for"]))
    print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
    max_id = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters

    print("Max ID", max_id)
    print("Dataset size", dataset_size)

    [encoded] = numpy.array(tokenizer.texts_to_sequences([text])) - 1

    train_size = dataset_size * 90 // 100
    dataset = create_char_level_dataset(encoded[:train_size], max_id)

    print()


def create_char_level_dataset(encoded, max_id):
    dataset = tensorflow.data.CharDataset.from_tensor_slices(encoded)

    n_steps = 100
    window_length = n_steps + 1  # target = input shifted 1 character ahead
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))

    numpy.random.seed(42)
    tensorflow.random.set_seed(42)

    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

    dataset = dataset.map(
        lambda X_batch, Y_batch: (tensorflow.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)

    for X_batch, Y_batch in dataset.take(1):
        print(X_batch.shape, Y_batch.shape)

    return dataset


def load_word_level_dataset():
    filepath = '/Users/fpena/Datasets/code-docstring-corpus/parallel_bodies'
    with open(filepath) as f:
        text = f.read()
        text = text.split("\n")
        text = text[:10000]
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(text)

    print(tokenizer.texts_to_sequences(["for"]))
    print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
    max_id = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters

    print("Max ID", max_id)
    print("Dataset size", dataset_size)

    encoded = numpy.array(tokenizer.texts_to_sequences(text))
    encoded = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(text), padding='post', dtype='float64')
    print(encoded.shape)
    encoded = numpy.reshape(encoded, (encoded.shape[0], 1, encoded.shape[1]))
    batch_size = 32

    print(encoded.shape)

    # print(encoded)

    train_size = dataset_size * 90 // 100

    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[1, encoded.shape[2]],
                         # dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        # keras.layers.GRU(128, return_sequences=True,
        #                  # dropout=0.2, recurrent_dropout=0.2),
        #                  dropout=0.2),
        # keras.layers.TimeDistributed(keras.layers.Dense(encoded.shape[2], activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(encoded, encoded, steps_per_epoch=train_size // batch_size, epochs=1)


def hola():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_words = 1500
    filepath = '/Users/fpena/Datasets/code-docstring-corpus/parallel_bodies'
    with open(filepath) as f:
        text = f.read()
        text = text.split("\n")
        text = text[:10000]
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, maxlen=32)

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, InputLayer

    numero_clases = 5

    modelo_sentimiento = Sequential()
    modelo_sentimiento.add(InputLayer(input_shape=(None, 32)))
    modelo_sentimiento.add(Embedding(max_words, 128, input_length=X.shape[1]))
    modelo_sentimiento.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    modelo_sentimiento.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))

    modelo_sentimiento.add(Dense(numero_clases, activation='softmax'))
    modelo_sentimiento.compile(loss='categorical_crossentropy', optimizer='adam',
                               metrics=['acc'])
    print(modelo_sentimiento.summary())


def main():
    # load_char_level_dataset()
    # load_word_level_dataset()
    hola()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
