# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
import pickle
import sys

import NBmodel

NUM_ARGS = 4
BOOL_MODEL_FILE = "Boolean.model"
NTF_MODEL_FILE = "NTF.model"

def train_dir(directory, cls, bool_model, bool_feat, ntf_model, ntf_feat):
    for f in NBmodel.get_files(directory):
        bool_ex = NBmodel.munge_Boolean(f, bool_features)
        ntf_ex = NBmodel.munge_NTF(f, ntf_features)

        bool_model.train(bool_ex, cls)
        ntf_model.train(ntf_ex, cls)

def dump_obj(filename, obj):
    handle = open(filename, 'wb')
    pickle.dump(obj, handle)
    handle.close()

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: train.py spamdir hamdir bool_features ntf_features"
        sys.exit(1)
    spamdir = sys.argv[1]
    hamdir = sys.argv[2]
    bool_features_file = sys.argv[3]
    ntf_features_file = sys.argv[4]

    bool_features = pickle.load(open(bool_features_file, 'rb'))
    ntf_features = pickle.load(open(ntf_features_file, 'rb'))

    bool_model = NBmodel.NB_Boolean()
    ntf_model = NBmodel.NB_NTF()

    train_dir(spamdir, NBmodel.SPAM, bool_model, bool_features, ntf_model,
              ntf_features)
    train_dir(hamdir, NBmodel.HAM, bool_model, bool_features, ntf_model,
              ntf_features)

    bool_features.finalize_training()
    ntf_features.finalize_training()

    dump_obj(BOOL_MODEL_FILE, bool_model)
    dump_obj(NTF_MODEL_FILE, ntf_model)

