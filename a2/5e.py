#!/usr/bin/env python
# Franklin Hu, Sunil Pedapudi
# CS 194-10
# Assignment 2

# Script to find optimal 'c' value for a particular classifier
import os
import re
import subprocess
import sys
import time

import parallel

OUTPUT_DIR = "5e_data"
station = None

c_data = {}

def get_model_path(c):
  return "%s/svm_model_%s_c_%f" % (OUTPUT_DIR, station, c)

#def check_interval(low, high, anneal=0):
#  print "+ Checking interval: %f, %f" % (low, high)
#  low_acc = check_c(low)
#  high_acc = check_c(high)
#
#  if (high <= low) or (high - low < 0.001):
#    argmax = max([(low_acc, low), (high_acc, high)])
#    return argmax[1], argmax[0]
#
#  midpoint = low + (float(high) - low) / 2
#  mid_acc = check_c(midpoint)
#
#  if anneal == 0:
#    argmax = max([(low_acc, low), (mid_acc, midpoint), (high_acc, high)])
#    return argmax[1], argmax[0]
#
#  if low_acc == mid_acc:
#    if mid_acc == high_acc:
#      return check_both(low, midpoint, high, anneal-1)
#    elif mid_acc < high_acc:
#      return check_interval(midpoint, high, anneal-1)
#    elif mid_acc > high_acc:
#      return check_interval(low, midpoint, anneal-1)
#  elif low_acc < mid_acc:
#    if mid_acc <= high_acc:
#      return check_interval(midpoint, high, anneal-1)
#    else:
#      return check_both(low, midpoint, high, anneal-1)
#  elif low_acc > mid_acc:
#    if mid_acc >= high_acc:
#      return check_interval(low, midpoint, anneal-1)
#    else:
#      return check_both(low, midpoint, high, anneal-1)
#  else:
#    raise Exception('Invalid code path: %f %f %f' % (low_acc, mid_acc, high_acc))
#
#def check_both(low, mid, high, anneal):
#  left_c, left_acc = check_interval(low, mid, anneal)
#  right_c, right_acc = check_interval(mid, high, anneal) 
#
#  if left_acc >= right_acc:
#    return left_c, left_acc
#  return right_c, right_acc
#
def check_c(c):
  if not os.path.exists(get_model_path(c)):
    do_learn(c)
  if c not in c_data:
    c_data[c] = do_classify(c)
  return c_data[c], c

def do_learn(c):
  print "++ Learning: %f" % c
  model_path = get_model_path(c)
  check_output(["svm_learn", "-c", str(c), data_file, model_path])

def do_classify(c):
  print "++ Classifying: %f" % c
  model_path = get_model_path(c)
  ret = check_output(["svm_classify", data_file, model_path, "/dev/null"])
  return get_accuracy(ret)

def check_output(args):
  return subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]

def get_accuracy(some_string):
  m = re.search('(?<=Accuracy on test set: )([0-9]+(\.[0-9]*)?)', some_string)
  if m:
    return float(m.group(0))
  raise Exception('No accuracy found in output')

def drange(start, stop, step):
  r = start
  while r < stop:
    yield (r)
    r += step

if __name__ == "__main__":
  if len(sys.argv) < 5:
    print "Usage: 5e.py data-file station low high"
    sys.exit(1)

  data_file = sys.argv[1]
  station = sys.argv[2]
  low = int(sys.argv[3])
  high = int(sys.argv[4])
  outfile = "optimal_c_station_%s_at_%d" % (station, time.time())

  #c, accuracy = check_interval(low, high, 10)
  #print c, accuracy

  inputs = [[10 ** x] for x in xrange(-3, 3)]
  workers = parallel.Workers()
  workers.initialize_n_workers(16)
  workers.set_function(check_c)
  workers.start()
  output = workers.run_over_data(inputs)

  handle = open(outfile, 'w')
  for accuracy, c in output:
    handle.write("c: %f\t%f\n" % (c, accuracy))
  handle.close()

