import math
import time

import numpy


def shuffle_parallel_files(file1, file2, postfix='_shuf'):

    with open(file1, encoding="ISO-8859-1") as f1:
        file1_lines = [line.rstrip() for line in f1]
    with open(file2, encoding="ISO-8859-1") as f2:
        file2_lines = [line.rstrip() for line in f2]

    assert len(file1_lines) == len(file2_lines), "'file1' and 'file2' must have the same number of lines"

    indices = numpy.arange(len(file1_lines))
    numpy.random.shuffle(indices)

    # print(indices)

    # file1_lines = file1_lines[indices]
    # file2_lines = file2_lines[indices]
    file1_lines = [file1_lines[i] for i in indices]
    file2_lines = [file2_lines[i] for i in indices]

    output_file1 = file1 + postfix
    output_file2 = file2 + postfix

    with open(output_file1, 'w') as f1:
        for item in file1_lines:
            f1.write("%s\n" % item)

    with open(output_file2, 'w') as f2:
        for item in file2_lines:
            f2.write("%s\n" % item)


def split_file(file_path, train_ratio, dev_ratio, test_ratio):

    numpy.testing.assert_almost_equal(train_ratio + dev_ratio + test_ratio, 1.0)

    with open(file_path, encoding="ISO-8859-1") as f:
        lines = [line.rstrip() for line in f]

    num_lines = len(lines)
    dev_size = math.floor(num_lines * dev_ratio)
    test_size = math.floor(num_lines * test_ratio)
    train_size = num_lines - dev_size - test_size

    assert len(lines) == train_size + dev_size + test_size

    train_file_path = file_path + '_train'
    dev_file_path = file_path + '_dev'
    test_file_path = file_path + '_test'

    with open(train_file_path, 'w') as f:
        for item in lines[:train_size]:
            f.write("%s\n" % item)

    with open(dev_file_path, 'w') as f:
        for item in lines[train_size:train_size+dev_size]:
            f.write("%s\n" % item)

    with open(test_file_path, 'w') as f:
        for item in lines[train_size + dev_size:]:
            f.write("%s\n" % item)


# TODO: Take this method out of here, move it to a data loader purpose module
def prepare_code_docstring_corpus_dataset():
    numpy.random.seed(42)
    code_snippets_file = '/Users/frape/Courses/Coursera-Deep-Learning/src/unif/data/parallel_bodies'
    descriptions_file = '/Users/frape/Courses/Coursera-Deep-Learning/src/unif/data/parallel_desc'
    # code_snippets_file = '/tmp/parallel_bodies'
    # descriptions_file = '/tmp/parallel_desc'
    shuffle_parallel_files(code_snippets_file, descriptions_file)
    split_file(code_snippets_file + '_shuf', 0.98, 0.01, 0.01)
    split_file(descriptions_file + '_shuf', 0.98, 0.01, 0.01)


def main():
    prepare_code_docstring_corpus_dataset()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
