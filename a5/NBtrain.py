# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
from collections import defaultdict
import math
import pickle
import random
import sys
import time

import NBmodel

NUM_ARGS = 3
NUM_RANDOM_FEATURES = 4000
BOOL_MODEL_FILE = "Boolean.model"
NTF_MODEL_FILE = "NTF.model"
GAMMA_MODEL_FILE = "Gamma.model"

def train_dir(directory, cls, model):
    exs = []
    for f in NBmodel.get_files(directory):
        exs.append(model.munge(f))
        if len(exs) > 500:
            for ex in exs:
                model.train(ex, cls)
            exs = []

def dump_obj(filename, obj):
    handle = open(filename, 'wb')
    pickle.dump(obj, handle)
    handle.close()

def _get_features_list(spam_dir, ham_dir):
    def process_email(freq, filename, cls):
        email = open(filename, 'r')
        for line in email:
            tok = line.rstrip("\n").rstrip("\r").strip(" ").split(" ")
            for t in tok:
                freq[(cls,t)] = [freq[(cls,t)][0]+1, 
                                 freq[(cls,t)][0]+float(1/len(tok))]
        email.close()

    freq = defaultdict(lambda: list([0,0]))
    for email_file in NBmodel.get_files(spam_dir):
        process_email(freq, email_file, 'SPAM')

    for email_file in NBmodel.get_files(ham_dir):
        process_email(freq, email_file, 'HAM')

    return freq
def _write_features(features_file, spam_dir, ham_dir, borntf):
    freq = _get_features_list(spam_dir, ham_dir)
    features = [x[1] for x in freq.keys()]
    infogain = []
    for f in features:
        sfreq, sig = freq[('SPAM', f)][borntf], 0
        hfreq, hig = freq[('HAM', f)][borntf], 0
        if sfreq > 0:
            sig = sfreq*math.log(sfreq)
        if hfreq > 0:
            hig = hfreq*math.log(hfreq)
        ig = -1*(sig+hig)
        infogain.append((ig, f))
    infogain.sort()
    features = [x[1] for x in infogain][:NUM_RANDOM_FEATURES]
    # random.shuffle(features)
    # features = features[:NUM_RANDOM_FEATURES]
    dump_obj(features_file, features)
    return features

def _write_bool_features(features_file, spam_dir, ham_dir):
    return _write_features(features_file, spam_dir, ham_dir, 0)

def _write_ntf_features(features_file, spam_dir, ham_dir):
    return _write_features(features_file, spam_dir, ham_dir, 1)

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: train.py spamdir hamdir features"
        sys.exit(1)
    spamdir = sys.argv[1]
    hamdir = sys.argv[2]
    features_file = sys.argv[3]

    gamma_model = NBmodel.NB_Gamma(features_file)

    t = time.time()

    print "+ Begin SPAM"
    train_dir(spamdir, NBmodel.SPAM, gamma_model)
    print "++ End SPAM %f" % (time.time() - t)

    t = time.time()
    print "+ Begin HAM"
    train_dir(hamdir, NBmodel.HAM, gamma_model)
    print "+ End HAM %f" % (time.time() - t)

    print "+ Dumping objects"
    dump_obj(GAMMA_MODEL_FILE, gamma_model)

