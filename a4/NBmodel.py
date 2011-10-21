# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
from collections import defaultdict
import math
import os
import pickle
import random

import numpy as np


#########################################
  
def munge_Boolean(email_file,features_file):
    """
    Returns a vector representation of the email based on features

    Arguments:
    email_file -- name of a preprocessed email file
    features_file -- name of a file containing pickled list of tokens
    """
    features = _get_features(features_file)
    freq, num_tokens = _munge_frequency(email_file)
    output = []
    for f in features:
        if freq[f] == 0:
            output.append(0)
        else:
            output.append(1)
    return output

def munge_NTF(email_file,features_file):
    """
    Returns a vector representation of the email based on features

    Arguments:
    email_file -- name of a preprocessed email file
    features_file -- name of a file containing pickled list of tokens
    """
    features = _get_features(features_file)
    freq, num_tokens = _munge_frequency(email_file)
    output = []
    for f in features:
        output.append(float(freq[f]) / num_tokens)
    return output

def _munge_frequency(email_file):
    """
    Counts the frequency of terms an email file
    """
    email = open(email_file, 'r')
    freq = defaultdict(int)
    num_tokens = 0
    for line in email:
        tok = line.strip(" ").split(" ")
        num_tokens += len(tok)
        for t in tok:
            freq[t] += 1
    email.close()

    return freq, num_tokens

def _get_features(features_file):
    handle = open(features_file, 'r')
    features = pickle.load(handle)
    handle.close()
    return features

def NBclassify_Boolean(example,model,cost_ratio):
    return 0

def NBclassify_NTF(example,model,cost_ratio):
    return 0


#########################################

HAM = 0
SPAM = 1

def get_files(path):
    for f in os.listdir(path):
        f = os.path.abspath( os.path.join(path, f ) )
        if os.path.isfile( f ):
            yield f

def get_empty_feature_list(n):
    return [0.0] * n

class NaiveBayesModel:
    
    #def __init__(self, features_file, model_file):
    #    self.features = pickle.load(open(features_file,'rb'))
    #    self.model = pickle.load(open(model_file,'rb'))
    def __init__(self, features_file):
        self.features = pickle.load(open(features_file, 'rb'))
        n = len(self.features)

        self.spam_data = (get_empty_feature_list(n), # partial sums
                          get_empty_feature_list(n), # partial variances
                          get_empty_feature_list(n)) # totals

        self.ham_data = (get_empty_feature_list(n), # partial sums
                          get_empty_feature_list(n), # partial variances
                          get_empty_feature_list(n)) # totals

        self.spams = 0.0
        self.hams = 0.0
        self.total = 0.0
    
    def train(self, example, cls):
        assert len(example) == len(self.features)
        if cls == SPAM:
            partial_sum, partial_var, total = self.spam_data
            self.spams += 1
        elif cls == HAM:
            partial_sum, partial_var, total = self.ham_data
            self.hams += 1
        else:
            raise Exception ('Non SPAM/HAM cls: %s' % cls)

        for i in xrange(len(example)):
            partial_sum[i] += example[i]
            partial_var[i] += example[i] ** 2
            total[i] += 1

        self.total += 1

    def finalize_training(self):
        n = len(self.features)
        self.spam_thetas = get_empty_feature_list(n)
        self.spam_vars = get_empty_feature_list(n)

        self.ham_thetas = get_empty_feature_list(n)
        self.ham_vars = get_empty_feature_list(n)

        spam_sum, spam_var, spam_total = self.spam_data
        ham_sum, ham_var, ham_total = self.ham_data

        for i in xrange(len(self.features)):
            self.spam_thetas[i] = spam_sum[i] / spam_total[i]
            self.spam_vars[i] = spam_var[i] / spam_total[i]

            self.ham_thetas[i] = ham_sum[i] / ham_total[i]
            self.ham_vars[i] = ham_var[i] / ham_total[i]

        self.finalized = True

    #def train(self, example, cls):
    #    raise Exception('UNIMPLEMENTED')

    def spam_probability(self, example):
        assert len(self.features) == len(example)
        attribute_probabilities = self._attribute_probabilities()
        return self.alpha * (self.spam_examples / self.total_examples) * \
                sum(attribute_probabilities)

    def _attribute_probabilities(self):
        sc = self.spam_counters
        tc = self.total_counters
        return [sc[i] / tc[i] if not tc[i] == 0 else 0
                for i in xrange(len(sc))]

    def test(self, spam_dir, ham_dir, cost_ratio):
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==1):
                loss += 1
    
        for f in get_files(ham_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==0):
                loss += cost_ratio
        
        print "Classifier average loss: %f" % (loss/N)


class NB_Boolean(NaiveBayesModel):

    def classify(self,example,cost_ratio):
        if not self.finalized:
            self.finalize_training()
        ratio = get_log_ratio(example)
        if ratio > 1:
            return SPAM
        elif ratio == 1:
            return random.choice([SPAM, HAM])
        else: # ratio < 1
            return HAM
        #return NBclassify_Boolean(example,self.model,cost_ratio)

    def get_log_ratio(self, example):
        return (math.log(self.spams) + \
                self.get_log_probability(self.spam_thetas, example)) / \
               (math.log(self.hams) + \
                self.get_log_probability(self.ham_thetas, example))

    def get_log_probability(self, theta_list, example):
        return sum([example[i] * math.log(theta_list[i]) + \
                    (1 - example[i]) * math.log(1 - theta_list[i]) \
                    for i in xrange(len(theta_list))])
        
    def munge(self,email_file):
        return munge_Boolean(email_file,self.features)


class NB_NTF(NaiveBayesModel):

    def __init__(self, features_file, b):
        NaiveBayesModel.__init__(self, features_file)
        self.b = b

    def _attribute_probabilities(self):
        attribute_probabilities = NaiveBayesModel._attribute_probabilities(
                self)
        return [exp_dist(self.b, x) for x in attribute_probabilities]
    
    def classify(self,example,cost_ratio):
        return NBclassify_NTF(example,self.model,cost_ratio)
    
    def munge(self,email_file):
        return munge_NTF(email_file,self.features)


#########################################
