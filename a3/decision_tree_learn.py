# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
import csv
from collections import defaultdict
import math
import pickle
import sys

from decision_tree import *

GOAL = 'goal'
STATIONS = []

def decision_tree_learn(examples, attributes, parent_examples):
  # If no examples, use pluraity of parent
  if not examples:
    return plurality_value(parent_examples)

  # If all are of the same class, use that class
  common_class = same_class(attribute, examples):
  if common_class:
    return common_class

  # If there are no more attributes left, use plurality
  if not attributes:
    return plurality_value(examples)

  A = get_best_attribute(attributes, examples)
  tree = decision_tree(A)

  for exs in group_by_attribute(A, examples):
    subtree = decision_tree_learn(exs, list(attribtues).remove(A), examples)
    tree.add_subtree(A, subtree)
  return tree

def plurality_value(examples):
  counts = get_goal_counts(examples)
  best_attr, best_val = get_plurality(counts)
  return decision_tree_leaf(best_attr)

def get_plurality_fraction(examples):
  counts = get_goal_counts(examples)
  best_attr, best_val = get_plurality(counts)
  return float(best_val) / len(examples)

def get_goal_counts(examples):
  if not examples:
    raise Exception('get_goal_counts called on empty examples list')

  counts = defaultdict(int)
  for e in examples:
    counts[e[GOAL]] += 1
  return counts

def get_plurality(counts):
  best_attr = None
  best_val = 0
  for k in counts:
    if counts[k] > best_val:
      best_attr = k
      best_val = counts[k]
  return best_attr, best_val

def same_class(attribute, examples):
  attr_val = None
  for e in examples:
    if attr_val is None:
      attr_val = e[attribute]
    elif not e[attribute] == attr_val:
      return False

  # All the same so create a leaf node with the common goal
  return decision_tree_leaf(attr_val)

def get_best_attribute(attributes, examples):
  best_attr = None
  best_gain = 0
  for a in attributes:
    gain = get_info_gain(a, examples)
    if gain > best_gain:
      best_attr = a
      best_gain = gain
  return best_attr

def get_info_gain(attribute, examples):
  total = len(examples)
  bucketed = bucket_examples_by_attribute(attribute, examples)

  remainder = get_remainder_over_buckets(bucketed, total)
  
  base_info = get_plurality_fraction(examples)
  return base_info - remainder

def bucket_examples_by_attribute(attribute, examples):
  bucketed = defaultdict(list)
  for e in examples:
    bucketed[e[attribute]].append(e)
  return bucketed

def get_remainder_over_buckets(bucketed, total_count):
  remainder = 0
  keys = bucketed.keys()
  for k in keys:
    elems = bucketed[k]
    weight = float(len(elems)) / total_count
    q = get_plurality_fraction(elems)
    remainder += weight * bool_entropy(q)
  return remainder

def bool_entropy(q):
  return -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: decision_tree_learn.py input-file"
    sys.exit(1)

  input_file = sys.argv[1]
  data = csv.reader(open(input_file, 'r'))
  examples = defaultdict(list)
  for datum in data:
    examples[datum[STATION_INDEX]].append(datum)
  
  for station in STATIONS:
    e = examples[station]
    dt = decision_tree_learn(examples, attributes, examples)
    dt.prune()
    outfile = "tree_%s" % station
    handle = open(outfile, 'w')
    handle.write(pickle.dump(dt))
    handle.close()


