# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011

# Implementation of internal nodes of a decision tree
class decision_tree:
  def __init__(self, attribute):
    self.attribute = attribute
    self.value_to_tree = {}

  def predict(self, example):
    pass

# Implementation of leaf nodes of a decision tree
class decision_tree_leaf(decision_tree):
  def __init__(self, value):
    self.value = value

  def predict(self, example):
    return self.value
