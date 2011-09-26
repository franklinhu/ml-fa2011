# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
import csv
from collections import defaultdict
from decision_tree import decision_tree
import pickle
import sys

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

  A = importance(attributes, examples)
  tree = decision_tree(A)

  raise Exception("NOT FINISHED")

def plurality_value(examples):
  if not examples:
    raise Exception('plurality_value called on empty examples list')

  counts = defaultdict(int)
  for e in examples:
    counts[GOAL] += 1

  attr_vals = counts.keys()
  attr_vals.sort(cmp=lambda x,y: cmp(counts[x], counts[y]), reverse=True)
  return attr_vals[0]

def same_class(attribute, examples):
  attr_val = None
  for e in examples:
    if attr_val is None:
      attr_val = e[attribute]
    elif not e[attribute] == attr_val:
      return False

  # All the same so create a leaf node with the common goal
  return decision_tree_leaf(attr_val)

def importance(attributes, examples):
  pass

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


