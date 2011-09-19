#!/usr/bin/env python
# Franklin Hu, Sunil Pedapudi
# CS 194-10
# Assignment 2
# gen-filtered-top-station-data

from collections import defaultdict
import csv
import random
import subprocess
import sys

DATA_DIR = "5fg_data"
MODEL_DIR = "5fg_models"

STATION_INDEX = 12
PHASE_INDEX = 28
PHASES = ['Lg', 'PKP', 'P', 'S', 'Rg', 'Sn', 'Pn', 'Pg']
TOP_STATIONS = [str(x) for x in [7, 24, 3, 80, 19, 38, 63, 12, 74, 65]]
FEATURE_INDICES = [
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

def transform_datum(datum, phase):
  features = []
  for i in xrange(len(FEATURE_INDICES)):
    name,idx = FEATURE_INDICES[i]
    features.append("%d:%s" % (i+1, datum[idx]))

  if datum[PHASE_INDEX].startswith(phase):
    target = 1
  else:
    target = -1
  return "%d %s" % (target, " ".join(features))

def station_outfiles(station, phase):
  everything = "%s/station_%s_%s.txt" % (DATA_DIR, station, phase)
  training = "%s/station_%s_%s_training.txt" % (DATA_DIR, station, phase)
  holdout = "%s/station_%s_%s_holdout.txt" % (DATA_DIR, station, phase)
  return open(everything, 'w'), open(training, 'w'), open(holdout, 'w')

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: gen-filtered-top-station-data <data-file>"
    sys.exit(1)

  filename = sys.argv[1]

  handle = open(filename, 'r')
  first_row = handle.next()
  data = csv.reader(handle)
  buffers = defaultdict(lambda : defaultdict(list))
  for datum in data:
    if not datum:
      continue

    station = datum[STATION_INDEX]
    for phase in PHASES:
      transformed = transform_datum(datum, phase)
      if station in TOP_STATIONS:
        buffers[station][phase].append(transformed)

  for station in buffers:
    for phase in buffers[station]:
      everything, training, holdout = station_outfiles(station, phase)
      for row in buffers[station][phase]:
        everything.write(row)
        everything.write("\n")

        if random.random() < 0.1:
          holdout.write(row)
          holdout.write("\n")
        else:
          training.write(row)
          training.write("\n")
      everything.close()
      training.close()
      holdout.close()

  for station in TOP_STATIONS:
    for phase in PHASES:
      print "+ SVM Learn: %s %s" % (station, phase)
      subprocess.Popen(["svm_learn", 
                        "%s/station_%s_%s_training.txt" % (DATA_DIR, 
                                                           station, phase),
                        "%s/model_%s_%s" % (MODEL_DIR, station, phase)])
