# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
import csv
from collections import defaultdict
import math
import pickle
import random
import sys

import dt_util
from decision_tree import *
import parallel

ATTRIBUTE_INDICES = [
  ('ddet60', 58),
  ('dtime60', 59), 
  ('hmxmn', 39), 
  ('htov0.25', 48), 
  ('htov0.5', 49), 
  ('htov1', 50), 
  ('htov2', 51), 
  ('htov4', 52), 
  ('hvrat', 38), 
  ('hvratp', 37), 
  ('inang1', 44), 
  ('inang3', 40), 
  ('per', 8), 
  ('plans', 35), 
  ('rect', 34), 
  ('arrival_slow', 4), 
  ('ddet100', 62), 
  ('dtime100', 63), 
  ('ddet300', 66), 
  ('dtime300', 67)]

GOAL = 'phase'
GOAL_INDEX = 28
STATIONS = [7, 24, 3, 80, 19, 38, 63, 12, 74, 65]
STATION_INDEX = 12
WORKERS = None

BAGGING_M = 10

def decision_tree_learn(examples, attributes, parent_examples):
  # If no examples, use plurality of parent
  if not examples:
    print "++ No examples, using plurality of parent"
    return dt_util.get_plurality_node(parent_examples)

  # If all are of the same class, use that class
  common_class = dt_util.same_class(examples)
  if common_class:
    print "++ All of common class"
    return common_class

  # If there are no more attributes left, use plurality
  if not attributes:
    print "++ No attributes, using plurality"
    return dt_util.get_plurality_node(examples)

  print "++ Computing best attribute"
  A, tree = get_best_attribute(attributes, examples)

  if A is None:
    return dt_util.get_plurality_node(examples)

  print "++ Splitting on attribute: %s" % A
 
  # Since all the attributes are continuous, do binary split
  lt_exs, gt_exs = split_by_attribute(A, tree.get_split_point(), examples)

  if (len(lt_exs) == 0) or (len(gt_exs) == 0):
    return dt_util.get_plurality_node(examples)

  lt_subtree = decision_tree_learn(lt_exs, attributes, examples)
  gt_subtree = decision_tree_learn(gt_exs, attributes, examples)

  tree.add_less_than_subtree(lt_subtree)
  tree.add_greater_than_subtree(gt_subtree)
  return tree

def get_best_attribute(attributes, examples):
  inputs = [[a, examples] for a in attributes]
  output = WORKERS.run_over_data(inputs)

  gain, best_attr, best_split_point = max(output)

  return best_attr, continuous_decision_tree(best_attr, best_split_point)

def get_attribute_info_gain(attribute, examples):
  gain, split_point = continuous_info_gain(attribute, examples)
  return gain, attribute, split_point

def continuous_info_gain(attribute, examples):
  sortable = list(examples)
  sortable.sort(cmp=lambda x,y: cmp(x.get(attribute), y.get(attribute)))

  index = 0
  total = len(sortable)
  best_gain = 0
  best_index = index + 1
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
    l_entropy = bool_entropy(dt_util.get_plurality_fraction(left))
    r_entropy = bool_entropy(dt_util.get_plurality_fraction(right))
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
  
  base_info = dt_util.get_plurality_fraction(examples)
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
    q = dt_util.get_plurality_fraction(elems)
    remainder += weight * bool_entropy(q)
  return remainder

def group_by_attribute(attribute, examples):
  buckets = defaultdict(list)
  for e in examples:
    buckets[e.get(attribute)].append(e) 
  return [(key, buckets[key]) for key in buckets]

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
    return 0
  retval = -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))
  if retval < 0:
    raise Exception(q)
  return retval

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
    examples[int(datum[STATION_INDEX])].append(dt_util.Example(datum))

  attributes = [a[1] for a in ATTRIBUTE_INDICES]

  WORKERS = parallel.Workers()
  WORKERS.initialize_n_workers(3)
  WORKERS.set_function(get_attribute_info_gain)
  WORKERS.start()
  
  for station in STATIONS:
    print "+ Station %d" % station
    exs = examples[station]

    # Generate decision tree
    """
    dt = decision_tree_learn(exs, attributes, exs)

    print "+ Pruning %d" % station
    dt.prune()
    outfile = "tree_%s" % station
    handle = open(outfile, 'w')
    pickle.dump(dt, handle)
    handle.close()
    """

    # Generate bagged decision tree
    trees = []
    for m in xrange(BAGGING_M):
      bag_exs = []
      for n in xrange(len(exs)):
        bag_exs.append(random.choice(exs))

      tree = decision_tree_learn(bag_exs, attributes, bag_exs)
      trees.append(tree)
    bagged_tree = bagged_decision_tree(trees)
    outfile = "bagged_tree_%s" % station
    handle = open(outfile, 'w')
    pickle.dump(bagged_tree, handle)
    handle.close()
