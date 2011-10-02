# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Assignment 3
import csv
import pickle
import sys

from decision_tree import *
import dtclassify

PHASE_INDEX = 28

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: test_decision_tree.py input_file decision_tree"
    sys.exit(1)
  input_file = sys.argv[1]
  decision_tree_file = sys.argv[2]

  dtfile = open(decision_tree_file)
  dt = pickle.load(dtfile)

  handle = csv.reader(open(input_file))
  handle.next() # Skip the first line

  success = 0
  total = 0
  for line in handle:
    if len(line) == 0:
      continue
    total += 1
    output = dtclassify.classify(dt, line)
    if output == line[PHASE_INDEX]:
      success += 1

  print success, total, float(success) / total
