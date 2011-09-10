from numpy import *
from numpy import matrix
from numpy import linalg
import numpy as np
import csv
import scipy.stats
import matplotlib.pyplot as plt

LAMBDA = 0.001

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
def klocalLinearRegressionFilteredData(station, phase, x, data, k, weight_function=None):
    projected_data = []
    max_dist = 0
    station_loc = None

    for d in data:
        if station_loc == None:
            station_loc = (float(d[6]), float(d[7]))
        loc = (float(d[1]), float(d[2]))
        if not weight_function == None:
            weight = math.sqrt(weight_function(0.1, loc, x))
        else:
            weight = 0
        projected_data.append((dist(loc, x), loc, float(d[10]), weight))

    sorted_list = sorted(projected_data, cmp=lambda a,b: cmp(a[0], b[0]))

    X = []
    Y = []
    weight_vector = []
    for elem in sorted_list[0:k]:
        X.append((1, elem[1][0], elem[1][1]))
        Y.append(elem[2])
        if not weight_function == None:
            weight_vector.append(elem[3])

    # Special case k=1 since we'll only have one sample
    #if k == 1:
    #    X.append((1, 0, 0))
    #    Y.append(0)

    if weight_function == None:
        weight_vector = None

    try:
        w_hat = getWHat(X, Y, False, weight_vector)
    except linalg.LinAlgError as e:
        w_hat = getWHat(X, Y, True, weight_vector)
    w_hat_transpose = transpose(w_hat)

    estimate = dot(w_hat_transpose, (1, x[0], x[1]))
    variance = getTrainingVariance(w_hat_transpose, data) #, getVariance(X, (1, x[0], x[1]))

    return estimate, variance

def getWHat(X, Y, useLambda, weights=None):
    if not weights == None:
        X_transpose = transpose(dot(weights * eye(len(weights)), X))
        Y = dot(weights, Y)
    else:
        X_transpose = transpose(X)
    tmp = dot(X_transpose, X)
    if useLambda:
        lambda_matrix = LAMBDA * eye(len(X[0]))
        tmp = tmp + lambda_matrix
    return dot(dot(linalg.inv(tmp), X_transpose),
               Y)

def getTrainingVariance(w_hat_transpose, data):
    errors = []
    for d in data:
        estimate = dot(w_hat_transpose, (1, float(d[1]), float(d[2])))
        actual = float(d[10])
        error = abs(estimate - actual)
        errors.append(error)
    return var(errors)

def getVariance(X, x_new):
   return dot(dot(transpose(x_new), 
                  linalg.inv(dot(transpose(X), X))),
              x_new)
                
def localLinearRegressionForP1(x, data):
    station = '1069'
    phase = 'P'
    k = 12
    filtered_data = filterData(station, phase, data)
    return klocalLinearRegressionFilteredData(station, phase, x, filtered_data, k)[0]

def localLinearRegressionForP2(x, data):
    station = '908'
    phase = 'P'
    k = 15
    filtered_data = filterData(station, phase, data)
    return klocalLinearRegressionFilteredData(station, phase, x, filtered_data, k)[0]

def localLinearRegressionForS1(x, data):
    station = '1069'
    phase = 'S'
    k = 8
    filtered_data = filterData(station, phase, data)
    return klocalLinearRegressionFilteredData(station, phase, x, filtered_data, k)[0]

def localLinearRegressionForS2(x, data):
    station = '908'
    phase = 'S'
    k = 58
    filtered_data = filterData(station, phase, data)
    return klocalLinearRegressionFilteredData(station, phase, x, filtered_data, k)[0]

## Estimate the residual time using locally weighted
## regression with Gaussian or Laplacian kernel
## Outputs estimate(float)
def localWeightedRegression(station, phase, x, data):
    filtered_data = filterData(station, phase, data)


def localWeightFunction1(b, x1, x2):
    return b/dist(x1, x2)

def localWeightFunction2(b, x1, x2):
    return e ** (-dist(x1, x2)/b)

def localWeightFunction3(b, x1, x2):
    return e ** (-dist(x1, x2) ** 2 / b ** 2)

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
    for i in xrange(len(data)/4):
       d = data[i]
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
    ks = xrange(12, 18, 1)

    variance_hit = 0
    variance_total = 0

    k_tested = []
    errors = []
    weighted_errors = []
    for i in xrange(factor):
        training,testing = pickCrossValidationBucket(bucketed_data, i)
        #k = ks[i]

        print "++ Testing fold %d with %d samples" % (i, len(testing))
        for k in ks:
            print "++ Test: k=%d" % k
            results = []
            #all_variances = []
            #weighted_results = []
            for test in testing:
                loc = (float(test[1]), float(test[2]))
                estimate,variance = klocalLinearRegressionFilteredData(station, phase, loc, training, k)
                actual = float(test[10])
                results.append((estimate, actual))
                #weighted_est, weighted_var = klocalLinearRegressionFilteredData(station, phase, loc, training, k, localWeightFunction2)
                #weighted_results.append((weighted_est, actual))

            	#all_variances.append(variance)

            #mean_var = sum(all_variances[0])/len(all_variances)
            #var_of_variances = sum([(var - mean_var)**2 for var in all_variances[0]])/len(all_variances)
            #stddev_of_variances = sqrt(var_of_variances)
            #for v in all_variances:
            #   if abs(v[0] - v[1]) <= stddev_of_variances:
            #      variance_hit += 1


            error = sum([(estimate-actual)**2 for estimate,actual in results])
            #weighted_error = sum([(estimate-actual)**2 for estimate, actual in weighted_results])
            print error #, weighted_error
            if error < 1000:
                k_tested.append(k) 
                errors.append(error) 

    plt.plot(k_tested, errors,'x')
    plt.show()

    #sums = {}
    #min_error = float('inf')
    best_k = 0
    #for e in errors:
    #    i, k, error = e
    #    if k in sums:
    #        sums[k] += error
    #    else:
    #       # sums[k] = error

    #    if sums[k] < min_error:
    #        best_k = k
    #        min_error = sums[k]

    print "Variance hits: %d/%d = %f" % (variance_hit, variance_total, float(variance_hit)/variance_total)

    return 0

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
    #print "1069 P: k=%d" % findBestKForLinearRegression('1069', 'P', data)
    #print "908  P: k=%d" % findBestKForLinearRegression('908', 'P', data)
    #print "1069 S: k=%d" % findBestKForLinearRegression('1069', 'S', data)
    #print "908  S: k=%d" % findBestKForLinearRegression('908', 'S', data)


