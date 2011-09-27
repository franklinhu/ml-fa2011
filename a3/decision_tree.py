# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011

# Implementation of internal nodes of a decision tree
class decision_tree:
  def __init__(self, attribute):
    self.attribute = attribute
    self.value_to_tree = {}

  def add_subtree(self, value, subtree):
    self.value_to_tree[value] = subtree

  def classify(self, example):
    value = example[self.attribute]
    return self.value_to_tree[value].classify(example)

class continuous_decision_tree:
  def __init__(self, attribute, split_point):
    self.attribute = attribute
    self.split_point = split_point

  def get_split_point(self):
    return self.split_point

  def add_less_than_subtree(self, subtree):
    self.lt_subtree = subtree

  def add_greater_than_subtree(self, subtree):
    self.gt_subtree = subtree

  def classify(self, example):
    value = example[self.attribute]
    if value < split_point:
      return self.lt_subtree.classify(example)
    return self.gt_subtree.classify(example)

# Implementation of leaf nodes of a decision tree
class decision_tree_leaf(decision_tree):
  def __init__(self, value):
    self.value = value

  def classify(self, example):
    return self.value
