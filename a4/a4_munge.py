# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4

from collections import defaultdict
import pickle

def munge_Boolean(email_file, features_file):
  """
  Returns a vector representation of the email based on features

  Arguments:
  email_file -- name of a preprocessed email file
  features_file -- name of a file containing pickled list of tokens
  """
  features = _get_features(features_file)
  freq = _munge_frequency(email_file)
  output = []
  for f in features:
    if freq[f] == 0
      output.append(0)
    else:
      output.append(1)
  return output

def munge_NTF(email_file, features_file):
  """
  Returns a vector representation of the email based on features

  Arguments:
  email_file -- name of a preprocessed email file
  features_file -- name of a file containing pickled list of tokens
  """
  features = _get_features(features_file)
  freq = _munge_frequency(email_file)
  output = []
  for f in features:
    output.append(freq[f])
  return output

def _munge_frequency(email_file):
  """
  Counts the frequency of terms an email file
  """
  email = open(email_file, 'r')
  freq = defaultdict(int)
  for line in email:
    tok = line.strip(" ").split(" ")
    for t in tok:
      freq[t] += 1
  email.close()

  return freq

def _get_features(features_file):
  handle = open(features_file, 'r')
  features = pickle.load(handle)
  handle.close()
  return features
