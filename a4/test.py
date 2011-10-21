# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
import pickle
import sys

import NBmodel

NUM_ARGS = 3

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: test.py model spamdir hamdir"
        sys.exit(1)

    model_file = sys.argv[1]
    spamdir = sys.argv[2]
    hamdir = sys.argv[3]

    model = pickle.load(open(model_file, 'r'))
    model.test(spamdir, hamdir, 1)

