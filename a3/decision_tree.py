# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011

from collections import defaultdict
import numpy as np
import scipy.stats

import dt_util

CONFIDENCE_BOUND = 100

# Implementation of internal nodes of a decision tree
class decision_tree:
  def __init__(self, attribute):
    self.attribute = attribute
    self.value_to_tree = {}

  def add_subtree(self, value, subtree):
    self.value_to_tree[value] = subtree

  def is_leaf(self):
    return False

  def prune(self):
    # Prune subtrees first
    for key in self.value_to_tree:
      subtree = self.value_to_tree[key]
      new_subtree = subtree.prune()
      self.value_to_tree[key] = new_subtree

    unsplit_counts = defaultdict(int)
    # Check that all children are leaves
    for subtree in self.value_to_tree.values():
      if not subtree.is_leaf():
        return self
      
      # Tally counts
      counts = subtree.get_counts()
      for key in counts:
        unsplit_counts[key] += counts[key]

    # Calculate chi-squared for children
    chi_square = sum([subtree.chi_square(unsplit_counts) \
                      for subtree in self.value_to_tree.values()])
    return self.evaluate_chi_square(chi_square, unsplit_counts)

  def evaluate_chi_square(self, chi_square, unsplit_counts):
    if chi_square <= CONFIDENCE_BOUND:
      return self
    
    print "++ Pruned"
    best_class, best_val = dt_util.get_plurality(unsplit_counts)
    return decision_tree_leaf(best_class, unsplit_counts)

  def classify(self, example):
    value = example.get(self.attribute)
    return self.value_to_tree[value].classify(example)


class continuous_decision_tree(decision_tree):
  def __init__(self, attribute, split_point):
    self.attribute = attribute
    self.split_point = split_point

  def get_split_point(self):
    return self.split_point

  def add_less_than_subtree(self, subtree):
    self.lt_subtree = subtree

  def add_greater_than_subtree(self, subtree):
    self.gt_subtree = subtree

  def prune(self):
    # Prune subtrees first
    self.lt_subtree = self.lt_subtree.prune()
    self.gt_subtree = self.gt_subtree.prune()

    # Check that children are leaves
    if (not self.lt_subtree.is_leaf()) or (not self.gt_subtree.is_leaf()):
      return self

    # Tally counts
    unsplit_counts = defaultdict(int)
    left_counts = self.lt_subtree.get_counts()
    right_counts = self.gt_subtree.get_counts()
    for key in left_counts:
      unsplit_counts[key] += left_counts[key]
    for key in right_counts:
      unsplit_counts[key] += right_counts[key]

    chi_square = self.lt_subtree.chi_square(unsplit_counts) +  \
                 self.gt_subtree.chi_square(unsplit_counts)

    return self.evaluate_chi_square(chi_square, unsplit_counts)

  def classify(self, example):
    value = example.get(self.attribute)
    if value < self.split_point:
      return self.lt_subtree.classify(example)
    return self.gt_subtree.classify(example)


# Implementation of leaf nodes of a decision tree
class decision_tree_leaf(decision_tree):
  def __init__(self, value, counts):
    self.value = value
    self.counts = counts

  def is_leaf(self):
    return True

  def get_counts(self):
    return self.counts

  def prune(self):
    return self

  def chi_square(self, expected_counts):
    observed = []
    expected = []
    for key in expected_counts:
      if key in self.counts:
        observed.append(self.counts[key])
      else:
        observed.append(0)
      expected.append(expected_counts[key])
    stat,p = scipy.stats.chisquare(np.array(observed), np.array(expected))
    return stat

  def classify(self, example):
    return self.value

class bagged_decision_tree:
  def __init__(self, trees):
    self.trees = trees

  def classify(self, example):
    predictions = [tree.classify(example) for tree in self.trees]
    cls, count = dt_util.get_plurality(dt_util.get_goal_counts(
                   predictions))
    return cls


