# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011

from collections import defaultdict
import numpy as np
import scipy.stats

import dt_util

CONFIDENCE_BOUND = 100

class dt_probability_leaf:
  """Decision tree leaf that returns the probability of an example
  when classifying given precomputed counts"""

  def __init__(self, counts, goal_index):
    self.counts = counts
    self.goal_index = goal_index
  
  def is_leaf(self):
    return True

  def get_counts(self):
    return self.counts

  def prune(self):
    return self

  def classify(self, example):
    val = example.get(self.goal_index)
    if val in self.counts:
      return float(self.counts[val]) / sum(self.counts.values())
    else:
      # Decision tree has never seen this before, so use smoothing
      return 1. / (sum(self.counts.values()) + 
                   len(self.counts.keys()) + 1)

# Implementation of internal nodes of a decision tree
class decision_tree(dt_probability_leaf):
  def __init__(self, attribute, counts, goal_index):
    dt_probability_leaf.__init__(self, counts, goal_index)
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
    if value in self.value_to_tree:
        return self.value_to_tree[value].classify(example)
    else:
        return dt_probability_leaf.classify(self, example)

