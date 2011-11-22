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

def decision_tree_learn(examples, attributes, parent_examples):
  # If no examples, use plurality of parent
  if not examples:
    #print "++ No examples, using probabilities of parent"
    return dt_util.get_probability_node(parent_examples)

  # If all are of the same class, use that class
  common_class = dt_util.same_class(examples)
  if common_class:
    #print "++ All of common class"
    return common_class

  # If there are no more attributes left, use plurality
  if not attributes:
    #print "++ No attributes, using plurality"
    return dt_util.get_probability_node(examples)

  #print "++ Computing best attribute"
  A, tree = get_best_attribute(attributes, examples)

  if A is None:
    return dt_util.get_probability_node(examples)
 
  for key,exs in group_by_attribute(A, examples):
    new_attr = list(attributes)
    new_attr.remove(A)
    subtree = decision_tree_learn(exs, new_attr, examples)
    tree.add_subtree(key, subtree)

  return tree

def get_best_attribute(attributes, examples):
  """
  workers = parallel.Workers()
  workers.initialize_n_workers(4)
  workers.set_function(get_attribute_info_gain)
  workers.start()

  inputs = [[a, examples] for a in attributes]
  output = workers.run_over_data(inputs)
  """
  best_gain = float("-inf")
  best_attr = None
  for a in attributes:
      gain, attr = get_attribute_info_gain(a, examples)
      if gain < best_gain:
          best_gain = gain
          best_attr = a

  return best_attr, decision_tree(best_attr, 
                                  dt_util.get_goal_counts(examples),
                                  dt_util.GOAL_INDEX)

def get_attribute_info_gain(attribute, examples):
  gain = discrete_info_gain(attribute, examples)
  return gain, attribute

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

def bool_entropy(q):
  if (q == 0) or (q == 1):
    return 0
  retval = -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))
  if retval < 0:
    raise Exception(q)
  return retval

def copy_without(l, item):
    new_l = list(l)
    new_l.remove(item)
    return new_l

def log_likelihood(example, trees):
    sum = 0
    for t in trees:
        sum += math.log(t.classify(example))
    return sum

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: decision_tree_learn.py input-file"
    sys.exit(1)

  input_file = sys.argv[1]
  data = csv.reader(open(input_file, 'r'))
  examples = defaultdict(list)
  examples = []

  # The first row of the CSV file is the schema (column names)
  schema = data.next()

  for datum in data:
    if not datum:
      continue
    examples.append(dt_util.Example(datum))
  random.shuffle(examples)

  # For A7 we use 2-fold cross validation
  pivot = len(examples) / 2
  set1 = examples[:pivot]
  set2 = examples[pivot:]
  print len(examples)

  all_attributes = 'GoodStudent,Age,VehicleYear,MakeModel,Mileage,'\
                   'Antilock,SeniorTrain,HomeBase,AntiTheft,PropCost,'\
                   'OtherCar,MedCost,Airbag,ILiCost,DrivHist'.split(',')
  num_attr = len(all_attributes)

  attributes = [
    (9, copy_without(range(num_attr), 9)),   # PropCost
    (13, copy_without(range(num_attr), 13)), # ILiCost
    (11, copy_without(range(num_attr), 11))  # MedCost
  ]

  set1_trees = []
  set2_trees = []

  for goal_index, attr in attributes:
    dt_util.GOAL_INDEX = goal_index

    # Create a decision tree for each set
    dt_1 = decision_tree_learn(set1, attr, set1)
    dt_2 = decision_tree_learn(set2, attr, set2)

    set1_trees.append(dt_1)
    set2_trees.append(dt_2)

  # Cross validate the likelihood of the other sets
  set1_likelihoods = sum(map(lambda ex: log_likelihood(ex, set1_trees), 
                             set2))
  set2_likelihoods = sum(map(lambda ex: log_likelihood(ex, set2_trees),
                             set1))
  print set1_likelihoods
  print set2_likelihoods
  print "Overall sum: %s" % (set1_likelihoods + set2_likelihoods)

