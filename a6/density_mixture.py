# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 6
import csv
import math
import sys

import cross_validation
import density_estimation

NUM_ARGS = 1
NUM_FOLDS = 5

def kernel_log_likelihood(example, fold_num, data, N):
    return density_estimation.log_kernel_density_a(data, example, 0, N,
                                                   fold_num)

def uniform_log_likelihood(example):
    return math.log(1. / (2 * (180 ** 2)))

def _usage():
    return "Usage: python density_mixture.py data_file"

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print _usage()
        sys.exit(1)

    data_file = sys.argv[1]
    handle = open(data_file, 'r')
    handle.readline()
    csv_file = csv.reader(handle)
    data = []

    for line in csv_file:
        l = tuple(map(lambda x: float(x), line[0:2]))
        data.append(l)

    weight_kernel = 1
    weight_uniform = 1
    cv = cross_validation.CrossValidation(NUM_FOLDS, data, True)
    for i in xrange(NUM_FOLDS):
        N = cv.num_training_examples(i)
        data = cv.training_examples
        for t in cv.training_examples(i):
            k_log_likelihood = kernel_log_likelihood(t, i, data, N)
            u_log_likelihood = uniform_log_likelihood(t)
            if k_log_likelihood > u_log_likelihood:
                weight_kernel += 1
            else:
                weight_uniform += 1
            print weight_kernel, weight_uniform, k_log_likelihood, u_log_likelihood

