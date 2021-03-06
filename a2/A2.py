# Franklin Hu and Sunil Pedapudi
# CS 194-10
# Assignment 2
# discrete-histogram
# confusion-matrix

from collections import defaultdict
import csv
import sys

class Counter:
  def __init__(self):
    self.counter = {}

  def increment(self, key):
    if key in self.counter:
      self.counter[key] += 1
    else:
      self.counter[key] = 1

  def get_all(self):
    return self.counter

def run_discrete_histogram():
  if len(sys.argv) < 3:
    print "Usage: discrete-histogram <data-file> <attribute>"
    sys.exit(1)
  filename = sys.argv[1]
  attribute = sys.argv[2]
  attribute_index = -1

  handle = open(filename, 'r')
  header = handle.next()
  header = header.rstrip("\n").split(",")

  # Get the index of the wanted attribute
  for i in xrange(len(header)):
    col = header[i]
    if col == attribute:
      attribute_index = i
      break
  
  # If attribute is not in list, abort
  if attribute_index == -1:
    print "Invalid attribute"
    sys.exit(2)

  # Count up instances of each value of attribute
  attribute_counter = Counter()
  data = csv.reader(handle)
  for line in data:
    if len(line) > 0:
      value = line[attribute_index]
      attribute_counter.increment(value)

  frequencies = attribute_counter.get_all()
  total = sum(frequencies.values())

  keys = frequencies.keys()
  keys.sort()
  for key in keys:
    print "%s,%s,%s" % (key,frequencies[key],float(frequencies[key])/total)

def run_confusion_matrix():
  if len(sys.argv) < 4:
    print "Usage: confusion-matrix <data-file> X Z"
    sys.exit(1)

  filename = sys.argv[1]
  X = sys.argv[2]
  Z = sys.argv[3]
  X_index = -1
  Z_index = -1

  handle = open(filename, 'r')
  header = handle.next()
  header = header.rstrip("\n").split(",")

  # Get the indices of the X and Z attributes
  for i in xrange(len(header)):
    col = header[i]
    if col == X:
      X_index = i
    elif col == Z:
      Z_index = i
  
  # If any of attributes are not in list, abor
  if X_index == -1:
    print "Invalid attribute: %s" % X
    sys.exit(2)
  if Z_index == -1:
    print "Invalid attribute: %s" % Z
    sys.exit(2)

  attribute_counter = defaultdict(lambda : defaultdict(int))
  data = csv.reader(handle)
  for line in data:
    if len(line) > 0:
      x_val = line[X_index]
      z_val = line[Z_index]
      attribute_counter[x_val][z_val] += 1

  print attribute_counter
  keys = attribute_counter.keys()
  for key in keys:
    print key, sum(attribute_counter[key].values())
