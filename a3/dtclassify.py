# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011

from decision_tree import *
from dt_util import *

def classify(decision_tree, example):
  e = Example(example)
  return decision_tree.classify(e)

def bagged_classify(bagged_decision_tree, example):
  return bagged_decision_tree.classify(Example(example))
