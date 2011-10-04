# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Assignment 3

from collections import defaultdict

# Wrapper class for training examples
class Example:
  def __init__(self, datum):
    self.data = datum
  
  def get(self, attribute_index):
    val = self.data[attribute_index]
    return float(val)

  def get_goal(self):
    return self.data[GOAL_INDEX]

# Returns dictionary with counts of the goal for each example 
# In this assignment the goal is 'phase'
def get_goal_counts(examples_list):
  if not examples_list:
    raise Exception('get_goal_counts called on empty examples list')

  counts = defaultdict(int)
  for e in examples_list:
    counts[e.get_goal()] += 1
  return counts

# Returns the plurality key and its value
def get_plurality(counts):
  best_attr = None
  best_val = 0
  for k in counts:
    if counts[k] > best_val:
      best_attr = k
      best_val = counts[k]
  return best_attr, best_val

# Returns the fraction out of total of plurality examples
def get_plurality_fraction(examples_list):
  counts = get_goal_counts(examples)
  best_attr, best_val = get_plurality(counts)
  return float(best_val) / len(examples)

# Returns a decision_tree_leaf with default classification as the
# plurality key
def get_plurality_node(examples_list):
  counts = get_goal_counts(examples_list)
  best_attr, best_val = get_plurality(counts)
  return decision_tree_leaf(best_attr, counts)

# Returns a decision_tree_leaf is all the examples are of the same class
# otherwise returns False
def same_class(examples):
  attr_val = None
  for e in examples:
    if attr_val is None:
      attr_val = e.get_goal()
    elif not e.get_goal() == attr_val:
      return False

  # All the same so create a leaf node with the common goal
  return decision_tree_leaf(attr_val, dt_util.get_goal_counts(examples))


