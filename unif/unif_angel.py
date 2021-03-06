import random
import subprocess
import sys
import time

import numpy as np

# subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
# from dcs_data_generator import DataGeneratorDCS
import tensorflow.keras as keras
import tables
import numpy as np
import random
import pickle5 as pickle
# from help import *
# from code_search_manager import CodeSearchManager


import subprocess
import sys

# try:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "pickle5"])
#     import pickle5 as pickle
# except:
#     import pickle

import tables
from tqdm import tqdm
# import numpy as np

def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size == -1:  # if chunk_size is set to -1, then, load all data
        chunk_size = data_len
    start_offset = start_offset % data_len
    sents = []
    for offset in tqdm(range(start_offset, start_offset + chunk_size)):
        offset = offset % data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents


def pad(data, len=None):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))


class DataGeneratorDCS(keras.utils.Sequence):
    def __init__(self, tokens_path, desc_path, batch_size, init_pos, last_pos, code_length, desc_length):
        self.tokens_path = tokens_path
        self.desc_path = desc_path
        self.batch_size = batch_size
        self.code_length = code_length
        self.desc_length = desc_length

        # code
        code_table = tables.open_file(tokens_path)
        self.code_data = code_table.get_node('/phrases')[:].astype(np.int)
        self.code_index = code_table.get_node('/indices')[:]
        self.full_data_len = self.code_index.shape[0]

        #self.full_data_len = 100 #100000

        # desc
        desc_table = tables.open_file(desc_path)
        self.desc_data = desc_table.get_node('/phrases')[:].astype(np.int)
        self.desc_index = desc_table.get_node('/indices')[:]

        self.init_pos = init_pos
        self.last_pos = min(last_pos, self.full_data_len)

        self.data_len = self.last_pos - self.init_pos
        print("First row", self.init_pos, "last row", self.last_pos, "len", self.__len__())

    def __len__(self):
        return (np.ceil((self.last_pos - self.init_pos) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        start_offset = idx * self.batch_size
        start_offset = start_offset % self.data_len
        chunk_size = self.batch_size

        code = []
        desc = []

        for offset in range(self.init_pos + start_offset, self.init_pos + start_offset + chunk_size):
            offset = offset % self.full_data_len

            # CODE
            len, pos = self.code_index[offset]['length'], self.code_index[offset]['pos']
            code.append(self.code_data[pos:pos + len].copy())

            # Desc
            len, pos = self.desc_index[offset]['length'], self.desc_index[offset]['pos']
            desc.append(self.desc_data[pos:pos + len].copy())

        code = self.pad(code, self.code_length)
        desc = self.pad(desc, self.desc_length)

        negative_description_vector = desc.copy()
        random.shuffle(negative_description_vector)

        results = np.zeros((self.batch_size, 1))

        return [np.array(code), np.array(desc), np.array(negative_description_vector)], results

    def test(self, idx):
        start_offset = idx * self.batch_size
        start_offset = start_offset % self.data_len
        chunk_size = self.batch_size

        code = []
        desc = []

        print(self.init_pos + start_offset, self.init_pos + start_offset + chunk_size)

        # return self.__getitem__(idx)

    def len(self):
        return self.__len__()

    def pad(self, data, len=None):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)


class CodeSearchManager():

    def get_dataset_meta(self):
        raise NotImplementedError(self)

    def get_dataset_meta_hardcoded(self):
        raise NotImplementedError()

    def generate_model(self):
        raise NotImplementedError()

    def load_weights(self, model, path):
        if os.path.isfile(path + '.index'):
            model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")

    def get_top_n(self, n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)

    def train(self, trainig_model, training_set_generator, weights_path, epochs=1):
        trainig_model.fit(training_set_generator, epochs=epochs)
        trainig_model.save_weights(weights_path)
        print("Model saved!")

    def test_embedded(self, dot_model, embedded_tokens, embedded_desc, results_path):

        results = {}
        pbar = tqdm(total=len(embedded_desc))

        for rowid, desc in enumerate(embedded_desc):
            expected_best_result = dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]

            deleted_tokens = np.delete(embedded_tokens, rowid, 0)

            tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

            prediction = dot_model.predict([deleted_tokens, tiled_desc], batch_size=32 * 4)

            results[rowid] = len(prediction[prediction > expected_best_result])

            pbar.update(1)
        pbar.close()

        top_1 = self.get_top_n(1, results)
        top_3 = self.get_top_n(3, results)
        top_5 = self.get_top_n(5, results)

        print(top_1)
        print(top_3)
        print(top_5)

        name = results_path + "/results-snnbert-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

        f = open(name, "a")

        f.write("top1,top3,top5\n")
        f.write( str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
        f.close()


class UNIF_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path

        # dataset info
        self.total_length = 18223872
        self.chunk_size = 100000   # 18223872  # 10000


        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading UNIF model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

    def get_dataset_meta_hardcoded(self):
        return 86, 410, 10001, 10001

    def get_dataset_meta(self):
        # 18223872 (len) #1000000
        code_vector = load_hdf5(data_path + "train.tokens.h5", 0, 18223872)
        desc_vector = load_hdf5(data_path + "train.desc.h5", 0, 18223872)
        vocabulary_merged = load_pickle(data_path + "vocab.merged.pkl")

        longer_code = max(len(t) for t in code_vector)
        print("longer_code", longer_code)
        longer_desc = max(len(t) for t in desc_vector)
        print("longer_desc", longer_desc)

        longer_sentence = max(longer_code, longer_desc)

        number_tokens = len(vocabulary_merged)

        return longer_sentence, number_tokens


    def generate_model(self, embedding_size, number_code_tokens, number_desc_tokens, code_length, desc_length, hinge_loss_margin):

        code_input = tf.keras.Input(shape=(code_length,), name="code_input")
        code_embeding = tf.keras.layers.Embedding(number_code_tokens, embedding_size, name="code_embeding")(code_input)

        attention_code = tf.keras.layers.Attention(name="attention_code")([code_embeding, code_embeding])

        query_input = tf.keras.Input(shape=(desc_length,), name="query_input")
        query_embeding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size, name="query_embeding")(
            query_input)

        code_output = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(attention_code)
        query_output = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")(query_embeding)

        # This model generates code embedding
        model_code = tf.keras.Model(inputs=[code_input], outputs=[code_output], name='model_code')
        # This model generates description/query embedding
        model_query = tf.keras.Model(inputs=[query_input], outputs=[query_output], name='model_query')

        # Cosine similarity
        # If normalize set to True, then the output of the dot product is the cosine proximity between the two samples.
        cos_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_sim')([code_output, query_output])

        # This model calculates cosine similarity between code and query pairs
        cos_model = tf.keras.Model(inputs=[code_input, query_input], outputs=[cos_sim], name='sim_model')

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(query_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                        name='dot_model')

        loss = tf.keras.layers.Flatten()(cos_sim)
        # training_model = tf.keras.Model(inputs=[ code_input, query_input], outputs=[cos_sim],name='training_model')

        model_code.compile(loss='cosine_proximity', optimizer='adam')
        model_query.compile(loss='cosine_proximity', optimizer='adam')

        cos_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])  # extract similarity

        # Negative sampling
        good_desc_input = tf.keras.Input(shape=(desc_length,), name="good_desc_input")
        bad_desc_input = tf.keras.Input(shape=(desc_length,), name="bad_desc_input")

        good_desc_output = cos_model([code_input, good_desc_input])
        bad_desc_output = cos_model([code_input, bad_desc_input])

        margin = 0.5
        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([good_desc_output, bad_desc_output])

        training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],
                                        name='training_model')

        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
        # y_true-y_true avoids warning

        return training_model, model_code, model_query, dot_model



    def test(self, model_code, model_query, dot_model, results_path, code_length, desc_length):
        test_tokens = load_hdf5(self.data_path + "test.tokens.h5" , 0, 100)
        test_desc = load_hdf5(self.data_path + "test.desc.h5" , 0, 100) # 10000

        test_tokens = pad(test_tokens, code_length)
        test_desc = pad(test_desc, desc_length)

        embedding_tokens = [None] * len(test_tokens)
        print("Embedding tokens...")
        for idx,token in enumerate(test_tokens):

            embedding_result = model_code(np.array(token).reshape(1,-1))
            embedding_tokens[idx] = embedding_result.numpy()[0]

        embedding_desc = [None] * len(test_desc)
        print("Embedding descs...")
        for idx,desc in enumerate(test_desc):

            embedding_result = model_query(np.array(desc).reshape(1,-1))
            embedding_desc[idx] = embedding_result.numpy()[0]

        self.test_embedded(dot_model, embedding_tokens, embedding_desc, results_path)



    def training_data_chunk(self, id, valid_perc):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid


    def load_dataset(self, data_chunk_id, batch_size):

        init_trainig, init_valid, end_valid = self.training_data_chunk(data_chunk_id, 0.8)

        longer_code, longer_desc, number_code_tokens, number_desc_tokens= self.get_dataset_meta_hardcoded()

        training_set_generator = DataGeneratorDCS(self.data_path + "train.tokens.h5", self.data_path + "train.desc.h5",
                                                  batch_size, init_trainig, init_valid, longer_code, longer_desc)
        return training_set_generator

if __name__ == "__main__":

    print("Running UNIF Model")

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    unif_dcs = UNIF_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 1

    dataset = unif_dcs.load_dataset(0, BATCH_SIZE)

    longer_code, longer_desc, number_code_tokens, number_desc_tokens= unif_dcs.get_dataset_meta_hardcoded()

    embedding_size = 2048

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            training_model, model_code, model_query, dot_model = unif_dcs.generate_model(embedding_size, number_code_tokens, number_desc_tokens, longer_code, longer_desc, 0.05)
            #unif_dcs.load_weights(training_model, script_path+"/../weights/unif_dcs_weights")
    else:
        training_model, model_code, model_query, dot_model = unif_dcs.generate_model(embedding_size, number_code_tokens,
                                                                                     number_desc_tokens, longer_code,
                                                                                     longer_desc, 0.05)
        #unif_dcs.load_weights(training_model, script_path + "/../weights/unif_dcs_weights")

    unif_dcs.train(training_model, dataset, script_path+"/../weights/unif_dcs_weights")

    unif_dcs.test(model_code, model_query, dot_model, script_path+"/../results", longer_code, longer_desc)


    unif_dcs.train(training_model, dataset, script_path+"/../weights/unif_dcs_weights")

    unif_dcs.test(model_code, model_query, dot_model, script_path+"/../results", longer_code, longer_desc)