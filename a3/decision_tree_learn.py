# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
import csv
from collections import defaultdict
import math
import pickle
import sys

from decision_tree import *

DISCRETE_ATTRIBUTES = [1, 9, 10, 13, 14, 27, 27, 47, 54]
GOAL = 'phase'
GOAL_INDEX = 28
STATIONS = [7, 24, 3, 80, 19, 38, 63, 12, 74, 65]
STATION_INDEX = 12

class Example:
  def __init__(self, datum):
    self.data = datum
  
  def get(self, attribute_index):
    val = self.data[attribute_index]
    if attribute_index in DISCRETE_ATTRIBUTES:
      return val
    return float(val)

  def get_goal(self):
    return self.data[GOAL_INDEX]

def decision_tree_learn(examples, attributes, parent_examples):
  print "+ Decision Tree Learn"
  # If no examples, use plurality of parent
  if not examples:
    return plurality_value(parent_examples)

  # If all are of the same class, use that class
  common_class = same_class(examples)
  if common_class:
    return common_class

  # If there are no more attributes left, use plurality
  if not attributes:
    return plurality_value(examples)

  print "++ Computing best attribute"
  A, tree = get_best_attribute(attributes, examples)

  print "++ Splitting on attribute: %s" % A
  if A in DISCRETE_ATTRIBUTES:
    for attr_val, exs in group_by_attribute(A, examples):
      subtree = decision_tree_learn(exs, list(attributes).remove(A), 
                                    examples)
      tree.add_subtree(attr_val, subtree)
    return tree
  
  # Since all the attributes are continuous, do binary split
  lt_exs, gt_exs = split_by_attribute(A, tree.get_split_point(), examples)
  lt_subtree = decision_tree_learn(lt_exs, attributes, examples)
  gt_subtree = decision_tree_learn(gt_exs, attributes, examples)

  tree.add_less_than_subtree(lt_subtree)
  tree.add_greater_than_subtree(gt_subtree)
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
    counts[e.get_goal()] += 1
  return counts

def get_plurality(counts):
  best_attr = None
  best_val = 0
  for k in counts:
    if counts[k] > best_val:
      best_attr = k
      best_val = counts[k]
  return best_attr, best_val

def same_class(examples):
  attr_val = None
  for e in examples:
    if attr_val is None:
      attr_val = e.get_goal()
    elif not e.get_goal() == attr_val:
      return False

  # All the same so create a leaf node with the common goal
  return decision_tree_leaf(attr_val)

def get_best_attribute(attributes, examples):
  best_attr = None
  best_gain = 0
  best_split_point = None
  for a in attributes:
    if a in DISCRETE_ATTRIBUTES:
      gain = discrete_info_gain(a, examples)
      split_point = None
    else:
      gain, split_point = continuous_info_gain(a, examples)
    
    print "+++ Tried attribute: %s %f" % (a, gain)
    if gain > best_gain:
      best_gain = gain
      best_attr = a
      best_split_point = split_point

  if best_attr in DISCRETE_ATTRIBUTES:
    return best_attr, decision_tree(best_attr)
  return best_attr, continuous_decision_tree(best_attr, best_split_point)

def continuous_info_gain(attribute, examples):
  sortable = list(examples)
  sortable.sort(cmp=lambda x,y: cmp(x.get(attribute), y.get(attribute)))

  index = 0
  total = len(sortable)
  best_gain = 0
  best_index = index
  while index < len(sortable):
    # Skip over elements that have the same class under this attribute
    current_goal = sortable[index].get_goal()
    while (index < len(sortable)) and \
          (current_goal == sortable[index].get_goal()):
      index += 1
    if index >= len(sortable):
      continue

    left = sortable[:index]
    right = sortable[index:]
    l_entropy = bool_entropy(get_plurality_fraction(left))
    r_entropy = bool_entropy(get_plurality_fraction(right))
    gain = (float(len(left)) / total) * l_entropy + \
           (float(len(right)) / total) * r_entropy
    if gain > best_gain:
      best_gain = gain
      best_index = index
  split_point = float(sum([e.get(attribute) for e in 
                          sortable[best_index-1:best_index+1]])) / 2

  return best_gain, split_point

def discrete_info_gain(attribute, examples):
  total = len(examples)
  bucketed = bucket_examples_by_attribute(attribute, examples)

  remainder = get_remainder_over_buckets(bucketed, total)
  
  base_info = get_plurality_fraction(examples)
  return base_info - remainder

def bucket_examples_by_attribute(attribute, examples):
  bucketed = defaultdict(list)
  for e in examples:
    bucketed[e.get(attribute)].append(e)
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

def group_by_attribute(attribute, examples):
  buckets = defaultdict(list)
  for e in examples:
    buckets[e.get(attribute)] = e
  return buckets.values()

def split_by_attribute(attribute, split_point, examples):
  lt_exs = []
  gt_exs = []
  for e in examples:
    if e.get(attribute) < split_point:
      lt_exs.append(e)
    else:
      gt_exs.append(e)
  return lt_exs, gt_exs

def bool_entropy(q):
  if (q == 0) or (q == 1):
    return -math.log(1, 2)
  return -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: decision_tree_learn.py input-file"
    sys.exit(1)

  input_file = sys.argv[1]
  data = csv.reader(open(input_file, 'r'))
  examples = defaultdict(list)

  # The first row of the CSV file is the schema (column names)
  schema = data.next()

  for datum in data:
    if not datum:
      continue
    examples[int(datum[STATION_INDEX])].append(Example(datum))

  attributes = range(len(schema))
  attributes.pop(GOAL_INDEX)
  
  for station in STATIONS:
    exs = examples[station]
    dt = decision_tree_learn(exs, attributes, exs)
    # dt.prune()
    outfile = "tree_%s" % station
    handle = open(outfile, 'w')
    handle.write(pickle.dump(dt))
    handle.close()


