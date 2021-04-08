import os
import time

import keras
import numpy
import tensorflow


class ShakespeareGenerator:

    def __init__(self):
        self.tokenizer = None
        self.max_id = 0
        self.dataset_size = 0

    def load_shakespeare(self):
        shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
        # shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        # filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
        filepath = '/Users/fpena/Datasets/code-docstring-corpus/parallel_bodies'
        with open(filepath) as f:
            shakespeare_text = f.read()
            shakespeare_text = shakespeare_text[:10000]
        self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(shakespeare_text)

        print(self.tokenizer.texts_to_sequences(["for"]))
        print(self.tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
        self.max_id = len(self.tokenizer.word_index)  # number of distinct characters
        self.dataset_size = self.tokenizer.document_count  # total number of characters

        print("Max ID", self.max_id)
        print("Dataset size", self.dataset_size)

        [encoded] = numpy.array(self.tokenizer.texts_to_sequences([shakespeare_text])) - 1

        train_size = self.dataset_size * 90 // 100
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
            lambda X_batch, Y_batch: (tensorflow.one_hot(X_batch, depth=self.max_id), Y_batch))
        dataset = dataset.prefetch(1)

        for X_batch, Y_batch in dataset.take(1):
            print(X_batch.shape, Y_batch.shape)

        return dataset

    def predict_next_char(self, model):
        X_new = self.preprocess(["How are yo"])
        # Y_pred = model.predict_classes(X_new)
        Y_pred = numpy.argmax(model.predict(X_new), axis=-1)
        print(self.tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])  # 1st sentence, last char

    def train_model(self, dataset):
        train_size = self.dataset_size * 90 // 100
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
        self.model = keras.models.Sequential([
            keras.layers.GRU(128, return_sequences=True, input_shape=[None, self.max_id],
                             # dropout=0.2, recurrent_dropout=0.2),
                             dropout=0.2),
            # keras.layers.GRU(128, return_sequences=True,
            #                  # dropout=0.2, recurrent_dropout=0.2),
            #                  dropout=0.2),
            keras.layers.TimeDistributed(keras.layers.Dense(self.max_id,
                                                            activation="softmax"))
        ])
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        history = self.model.fit(dataset, steps_per_epoch=train_size // batch_size,
                            epochs=1)

    def preprocess(self, texts):
        X = numpy.array(self.tokenizer.texts_to_sequences(texts)) - 1
        return tensorflow.one_hot(X, self.max_id)

    def next_char(self, text, temperature=1):
        X_new = self.preprocess([text])
        y_proba = self.model.predict(X_new)[0, -1:, :]
        rescaled_logits = tensorflow.math.log(y_proba) / temperature
        char_id = tensorflow.random.categorical(rescaled_logits, num_samples=1) + 1
        return self.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(self, text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += self.next_char(text, temperature)
        return text


def tokenizer_test():
    # source text
    data = """ Jack and Jill went up the hill\n
    		To fetch a pail of water\n
    		Jack fell down and broke his crown\n
    		And Jill came tumbling after\n """

    filepath = '/Users/fpena/Datasets/code-docstring-corpus/parallel_bodies'
    with open(filepath) as f:
        shakespeare_text = f.read()
        shakespeare_text = shakespeare_text[:1000000]

    # integer encode text
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0]
    # determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)


def main():
    shakespeare_generator = ShakespeareGenerator()
    dataset = shakespeare_generator.load_shakespeare()
    shakespeare_generator.train_model(dataset)
    # print(shakespeare_generator.next_char("How are yo", temperature=1))
    # print(shakespeare_generator.complete_text("t", temperature=1))
    # print(shakespeare_generator.complete_text("his", temperature=1))
    # print(shakespeare_generator.complete_text("the morning ", temperature=1))
    print(shakespeare_generator.next_char("retur", temperature=1))
    print(shakespeare_generator.complete_text("for", temperature=1))
    print(shakespeare_generator.complete_text("if", temperature=1))
    print(shakespeare_generator.complete_text("return  ", temperature=1))

    tokenizer_test()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
