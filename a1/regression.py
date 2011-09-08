from numpy import *
from numpy import matrix
from numpy import linalg
import numpy as np
import csv
import scipy.stats

LAMBDA = 0.1

## Calculates the great circle distance between two point on the earth's
## surface in degrees. loc1 and loc2 are pairs of longitude and latitude. E.g.
## int(dist_deg((10,0), (20, 0))) gives 10
def dist(loc1, loc2):
   DEG2RAD = np.pi / 180
   RAD2DEG = 180 / np.pi
   lon1, lat1 = loc1
   lon2, lat2 = loc2
   return np.arccos(np.sin(lat1 * DEG2RAD) * np.sin(lat2 * DEG2RAD)
                    + np.cos(lat1 * DEG2RAD) * np.cos(lat2 * DEG2RAD)
                    * np.cos((lon2 - lon1) * DEG2RAD)) * RAD2DEG

## Estimates the residual time of a query point x using local
## linear regression.
## Inputs: station(str), phase(str), x(list of 2 floats), data(cvs.reader object)
## E.g. localLinearRegression('908','P', [10.25,-15.50] , 6)
## Outputs estimate(float) and varestimate(float) using the return command, e.g.
## return estimate, varestimate
def klocalLinearRegression(station, phase, x, data, k):
    filtered_data = filterData(station, phase, data)
    return klocalLinearRegressionFilteredData(station, phase, x, filtered_data, k)

# Invariant: elements in data all are from station _station_ and 
#            phase _phase_
def klocalLinearRegressionFilteredData(station, phase, x, data, k):
    projected_data = []
    max_dist = 0
    station_loc = None
    for d in data:
        if station_loc == None:
            station_loc = (float(d[6]), float(d[7]))
        loc = (float(d[1]), float(d[2]))
        projected_data.append((dist(loc, x), dist(loc, station_loc), float(d[10])))

    sorted_list = sorted(projected_data, cmp=lambda a,b: cmp(a[0], b[0]))

    X = []
    Y = []
    for elem in sorted_list[0:k]:
        X.append((1, elem[1]))
        Y.append(elem[2])

    if len(X) == 0:
        print k, sorted_list
    w_hat = getWHat(X, Y)
    w_hat_transpose = transpose(w_hat)

    estimate = dot(w_hat_transpose, (1, dist(x, station_loc)))
    variance = -1

    return estimate, variance

def getWHat(X, Y):
    X_transpose = transpose(X)
    lambda_matrix = LAMBDA * eye(len(X[0]))
    return dot(dot(linalg.inv(dot(X_transpose, X) + lambda_matrix), 
                   X_transpose), 
               Y)
                
def localLinearRegressionForP1(x, data):
    station = '1069'
    return klocalLinearRegression(station, 'P', x, data, k)

def localLinearRegressionForP2(x, data):
    station = '908'
    return klocalLinearRegression(station, 'P', x, data, k)

def localLinearRegressionForS1(x, data):
    station = '1069'
    return klocalLinearRegression(station, 'S', x, data, k)

def localLinearRegressionForS2(x, data):
    station = '908'
    return klocalLinearRegression(station, 'S', x, data, k)

## Estimate the residual time using locally weighted
## regression with Gaussian or Laplacian kernel
## Outputs estimate(float)
def localWeightedRegression(station, phase, x, data):
   pass

def filterData(station, phase, data):
    print "++ Filtering for station %s with phase %s" % (station, phase)
    filtered_data = []
    for d in data:
        if (station == d[5]) and (phase == d[9]):
            filtered_data.append(d)
    return filtered_data

def bucketForCrossValidation(data, factor):
    print "++ Bucketing into %d buckets" % factor
    buckets = [[] for x in xrange(factor)]
    for d in data:
        r = random.random_integers(0, factor-1)
        buckets[r].append(d)
    return buckets
    
def pickCrossValidationBucket(bucketed_data, bucket_num):
    concat = []
    for b in bucketed_data[0:bucket_num]:
        concat += b
    for b in bucketed_data[bucket_num+1:]:
        concat += b
    return concat, bucketed_data[bucket_num]

def findBestKForLinearRegression(station, phase, data):
    print "+ Finding best k for station %s with phase %s" % (station, phase)
    factor = 10
    filtered_data = filterData(station, phase, data)
    bucketed_data = bucketForCrossValidation(filtered_data, factor)
    ks = range(1, factor+1)
    random.shuffle(ks)
    min_error = float('inf')
    best_k = 0
    for i in xrange(factor):
        training,testing = pickCrossValidationBucket(bucketed_data, i)
        k = ks[i]

        print "++ Test: k=%d on %d samples" % (k, len(testing))
        results = []
        for test in testing:
            loc = (float(test[1]), float(test[2]))
            estimate,variance = klocalLinearRegressionFilteredData(station, phase, loc, training, k)
            actual = float(test[10])
            results.append((estimate, actual))

        error = sum([(estimate-actual)**2 for estimate,actual in results])
        print "++ Error for k=%d: %f" % (k, error)
        if error < min_error:
            min_error = error
            best_k = k
    return best_k



## An example to read data and count the number of P-phase detections
## and do matrix operations
# data = csv.reader(open('trainingData.csv'));
# count = 0
# for event in data:
#         if event[9]=='P':
#            count = count + 1
# print count
# A = matrix( [[1,2,3],[11,12,13],[21,22,23]]) # Creates a matrix.
# print A.T                                    # Transpose of A.
# print A.I                                    # Inverse of A.

def add_event(event_dict, station_id):
  if station_id in event_dict:
      event_dict[station_id] += 1
  else:
      event_dict[station_id] = 1

def printTopStations(data):
    p_events = {}
    s_events = {}
    for event in data:
        if event[9] == 'P':
            add_event(p_events, event[5])
        elif event[9] == 'S':
            add_event(s_events, event[5])

    for station in sorted(p_events, key=p_events.get, reverse=True):
        print station, p_events[station]
    print ""
    for station in sorted(s_events, key=s_events.get, reverse=True):
        print station, s_events[station]

if __name__ == "__main__":
    handle = csv.reader(open('trainingData.csv'))
    data = []
    for row in handle:
        data.append(row)
    #printTopStations(data)

    print "1069, P, %s" % str(klocalLinearRegression('1069', 'P', (0, 0), data, 6))
    print "908, P, %s" % str(klocalLinearRegression('908', 'P', (0, 0), data, 6))
    print "1069, S, %s" % str(klocalLinearRegression('1069', 'S', (0, 0), data, 6))
    print "908, S, %s" % str(klocalLinearRegression('908', 'S', (0, 0), data, 6))

    print "Best k values:"
    print "1069 P: k=%d" % findBestKForLinearRegression('1069', 'P', data)
    print "908  P: k=%d" % findBestKForLinearRegression('908', 'P', data)
    print "1069 S: k=%d" % findBestKForLinearRegression('1069', 'S', data)
    print "908  S: k=%d" % findBestKForLinearRegression('908', 'S', data)


