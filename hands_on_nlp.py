import os
import time

import keras
import numpy
import tensorflow


def load_shakespeare_tokenizer():
    shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(shakespeare_text)

    return tokenizer


def load_shakespeare():
    shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
    # shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    print(tokenizer.texts_to_sequences(["First"]))
    print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
    max_id = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters

    print("Max ID", max_id)
    print("Dataset size", dataset_size)

    [encoded] = numpy.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

    train_size = dataset_size * 90 // 100
    dataset = tensorflow.data.CharDataset.from_tensor_slices(encoded[:train_size])

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


def predict_next_char(model, tokenizer):
    X_new = preprocess(["How are yo"], tokenizer)
    # Y_pred = model.predict_classes(X_new)
    Y_pred = numpy.argmax(model.predict(X_new), axis=-1)
    print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])  # 1st sentence, last char


def train_model(dataset, tokenizer):
    max_id = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters
    train_size = dataset_size * 90 // 100
    batch_size = 32

    for X_batch, Y_batch in dataset.take(1):
        print(X_batch.shape, Y_batch.shape)

    checkpoint_path = "/Users/fpena/tmp/tf_shakespeare/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # model = keras.models.Sequential([
    #     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    #     keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
    # ])
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    # history = model.fit(dataset, epochs=20)
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         # dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         # dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, steps_per_epoch=train_size // batch_size,
                        epochs=10)


def preprocess(texts, tokenizer):
    max_id = len(tokenizer.word_index)  # number of distinct characters
    X = numpy.array(tokenizer.texts_to_sequences(texts)) - 1
    return tensorflow.one_hot(X, max_id)


def main():
    tokenizer = load_shakespeare_tokenizer()
    dataset = load_shakespeare()
    # print(dataset)
    train_model(dataset, tokenizer)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
