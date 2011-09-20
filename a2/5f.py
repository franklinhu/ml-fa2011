from collections import defaultdict
import json
from pprint import pprint

stations = [12, 19, 24, 3, 38, 63, 65, 7, 74, 80]

if __name__ == "__main__":

  d = defaultdict(lambda : defaultdict(int))
  for station in stations:
    handle = open('station_%d_counters' % station)
    contents = handle.readlines()
    handle.close()
    contents = "".join(contents)

    data = json.loads(contents)
    for key1 in data:
      for key2 in data[key1]:
        d[key1][key2] += data[key1][key2]
  print d

