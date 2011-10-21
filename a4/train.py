# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
from collections import defaultdict
import pickle
import random
import sys
import time

import NBmodel

NUM_ARGS = 4
NUM_RANDOM_FEATURES = 3000
BOOL_MODEL_FILE = "Boolean.model"
NTF_MODEL_FILE = "NTF.model"

def train_dir(directory, cls, bool_model, bool_feat, ntf_model, ntf_feat):
    for f in NBmodel.get_files(directory):
        bool_ex = NBmodel.munge_Boolean(f, bool_feat)
        ntf_ex = NBmodel.munge_NTF(f, ntf_feat)

        bool_model.train(bool_ex, cls)
        ntf_model.train(ntf_ex, cls)

def dump_obj(filename, obj):
    handle = open(filename, 'wb')
    pickle.dump(obj, handle)
    handle.close()

def _write_features(features_file, spam_dir, ham_dir):
    freq = defaultdict(int)
    for email_file in NBmodel.get_files(spam_dir):
        email = open(email_file, 'r')
        for line in email:
            tok = line.strip(" ").split(" ")
            for t in tok:
                freq[('SPAM',t)] += 1
        email.close()
    for email_file in NBmodel.get_files(ham_dir):
        email = open(email_file, 'r')
        for line in email:
            tok = line.strip(" ").split(" ")
            for t in tok:
                freq[('HAM',t)] += 1
        email.close()

    features = [x[1] for x in freq.keys()]
    random.shuffle(features)
    features = features[:NUM_RANDOM_FEATURES]
    dump_obj(features_file, features)
    return features

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: train.py spamdir hamdir bool_features ntf_features"
        sys.exit(1)
    spamdir = sys.argv[1]
    hamdir = sys.argv[2]
    bool_features_file = sys.argv[3]
    ntf_features_file = sys.argv[4]

    features = _write_features(bool_features_file, spamdir, hamdir)
    bool_features = features
    ntf_features = features

    #bool_features = pickle.load(open(bool_features_file, 'rb'))
    #ntf_features = pickle.load(open(ntf_features_file, 'rb'))

    bool_model = NBmodel.NB_Boolean(bool_features_file)
    ntf_model = NBmodel.NB_NTF(bool_features_file, 0.1)

    t = time.time()

    print "+ Begin SPAM"
    train_dir(spamdir, NBmodel.SPAM, bool_model, bool_features, 
              ntf_model, ntf_features)
    print "++ End SPAM %f" % (time.time() - t)

    t = time.time()
    print "+ Begin HAM"
    train_dir(hamdir, NBmodel.HAM, bool_model, bool_features,
              ntf_model, ntf_features)
    print "+ End HAM %f" % (time.time() - t)

    t = time.time()
    print "+ Begin Boolean finalize"
    bool_model.finalize_training()
    print "+ %f" % (time.time() - t)

    t = time.time()
    print "+ Begin NTF finalize"
    ntf_model.finalize_training()
    print "+ %f" % (time.time() - t)
    
    print "+ Dumping objects"
    dump_obj(BOOL_MODEL_FILE, bool_model)
    dump_obj(NTF_MODEL_FILE, ntf_model)

