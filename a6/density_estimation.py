# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 6
import collections
import csv
import heapq
import math
import random
import sys

import numpy as np

import cross_validation

NUM_ARGS = 1
NUM_FOLDS = 5
CURRENT_FOLD = None

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

# Cache the following since computations are expensive
DISTANCE_CACHE = collections.OrderedDict()
KERNEL_CACHE = collections.OrderedDict()

## Calculates the great circle distance between two point on the earth's
## surface in degrees. loc1 and loc2 are pairs of longitude and latitude. E.g.
## int(dist_deg((10,0), (20, 0))) gives 10
def dist(loc1, loc2):
    if loc1 > loc2:
        tmp = loc1
        loc1 = loc2
        loc2 = tmp
    key = (loc1, loc2)
    if key not in DISTANCE_CACHE:
        # Check the size and evict 1/4 of the dict if size > 8 000 000
        if len(DISTANCE_CACHE) > 8000000:
            for i in xrange(len(DISTANCE_CACHE)/4):
                DISTANCE_CACHE.popitem(last=False)

        lon1, lat1 = loc1
        lon2, lat2 = loc2
        tmp = math.sin(lat1 * DEG2RAD) * math.sin(lat2 * DEG2RAD) \
                      + math.cos(lat1 * DEG2RAD) * math.cos(lat2 * DEG2RAD)\
                      * math.cos((lon2 - lon1) * DEG2RAD)
        # Floating point rounding issues for tmp = 1.0000000000...1
        # cause domain error when taking arccos, so we clamp it between -1,1
        tmp = min(max(tmp, -1), 1)
        retVal = math.acos(tmp) * RAD2DEG
        DISTANCE_CACHE[key] = retVal

    else:
        retVal = DISTANCE_CACHE[key]
        del DISTANCE_CACHE[key]
        DISTANCE_CACHE[key] = retVal
    return retVal

def dist_k(k, data, query_point):
    k_nearest = k_nearest_neighbors(k, data, query_point)
    k_nearest.sort()
    return k_nearest[k-1]

def k_nearest_neighbors(k, data, query_point):
    """
    Find the k closest points to the query_point, in no particular order
    """
    h = []
    for d in data(CURRENT_FOLD):
        distance = dist(query_point, d)
        if len(h) < k:
            heapq.heappush(h, -distance)
        heapq.heappushpop(h, -distance)
    return map(lambda x: -x, h)

def laplacian(d, b):
    return math.e ** (-d/b)
    """
    ratio = -d/b
    if ratio not in KERNEL_CACHE:
        KERNEL_CACHE[ratio] = math.e ** (-d/b)
    return KERNEL_CACHE[ratio]
    """

def kernel_density_base(kernel, distance, width, data, query_point, k, N):
    """
    Common function since 3/4 of the densities functions involve same code

    Arguments:
    kernel -- function (in this case Laplacian) that takes a distance d
              and width b parameters
    distance -- function that computes the distance between two points
    width -- function that computes the width b for the kernel function
    data -- list of training data entries
    query_point -- the point in question (x)
    """
    kernels = [kernel(distance(query_point, x_i),
                              width(k, data, x_i, query_point)) 
                       for x_i in data(CURRENT_FOLD)]
    kernel_sum = sum(kernels)
    try:
        retVal = math.log(kernel_sum) - math.log(N)
    except ValueError as e:
        print kernel_sum, N, len(kernels)
        raise Exception(e)
    return retVal

def log_kernel_density_a(data, query_point, k, N):
    """
    Kernel density function as given by:
    P(x) = (1/N) * \sum\limits_{i=1}^N K_b(d(x,x_i))
    """
    return kernel_density_base(laplacian, dist, 
                               lambda k,data,x_i,x: 5, 
                               data, query_point, k, N)

def log_kernel_density_b(data, query_point, k, N):
    """
    P(x) = k / (2N * d_k(x))
    where d_k(x) is the distance from x to the kth nearest neighbor of x
    """
    return math.log(k) - math.log(2 * N * dist_k(k, data, query_point))

def log_kernel_density_c(data, query_point, k, N):
    """
    P(x) = (1/N) * \sum\limits_{i=1}^N K_{d_k(x)}(d(x,x_i))
    same as (a) except the kernel width is determined by the kth
    nearest neighbor distance
    """
    return kernel_density_base(laplacian, dist,
                               lambda k,data,x_i,x: dist_k(k, data, x),
                               data, query_point, k, N)

def log_kernel_density_d(data, query_point, k, N):
    """
    P(x) = (1/N) * \sum\limits_{i=1}^N K_{d_{ik}(x)}(d(x,x_i))
    same as (c) except the kernel width is the kth nearest neighbor from 
    each x_i to its kth nearest neighbor, rather than from x
    """
    return kernel_density_base(laplacian, dist, 
                               lambda k,data,x_i,x: dist_k(k, data, x_i),
                               data, query_point, k, N)

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: density_estimation.py data_file"
        sys.exit(1)
    data_file = sys.argv[1]
    handle = open(data_file, 'r')
    handle.readline()
    csv_file = csv.reader(handle)
    data = []
    LON = -73.125
    DELTA = 15
    for line in csv_file:
        l = tuple(map(lambda x: float(x), line[0:2]))
        data.append(l)

    random.shuffle(data)

    # Find optimal k
    """
    NUM_FOLDS = 20
    cross_val = cross_validation.CrossValidation(NUM_FOLDS, data)
    densities = [
        log_kernel_density_a,
        log_kernel_density_b,
        log_kernel_density_c,
        log_kernel_density_d]
    ks = range(6, 6 + NUM_FOLDS)
    min_likelihoods = [float("inf")] * len(densities)
    for i in xrange(NUM_FOLDS):
        CURRENT_FOLD = i
        sum_likelihoods = [0] * len(densities)
        for v_ex in cross_val.validation_examples(i):
            N = cross_val.num_training_examples(i)
            likelihoods = map(lambda x: x(cross_val.training_examples,
                                          v_ex, ks[i], N),
                              densities)
            for j in xrange(len(densities)):
                sum_likelihoods[j] += likelihoods[j]

        sum_likelihoods = map(lambda x: x / \
            cross_val.num_validation_examples(i), sum_likelihoods)
        for i in xrange(len(sum_likelihoods)):
            min_likelihoods[i] = min(sum_likelihoods[i], min_likelihoods[i])
        print "Sum: ", sum_likelihoods
        print "Min: ", min_likelihoods
    """

    # use k = 8
    k = 15
    NUM_FOLDS = 5
    cv = cross_validation.CrossValidation(NUM_FOLDS, data)
    densities = [
        log_kernel_density_a#,
        #log_kernel_density_b,
        #log_kernel_density_c,
        #log_kernel_density_d
        ]
    for i in xrange(NUM_FOLDS):
        CURRENT_FOLD = i
        sum_likelihoods = [0] * len(densities)
        for v_ex in cv.validation_examples(i):
            N = cv.num_training_examples(i)
            log_likelihoods = map(lambda x: x(cv.training_examples,
                                          v_ex, k, N),
                              densities)
            likelihoods = map(lambda x: math.e ** x, log_likelihoods)
            for j in xrange(len(densities)):
                sum_likelihoods[j] += likelihoods[j]

        num_validation = cv.num_validation_examples(i)
        sum_likelihoods = map(lambda x: x / num_validation, sum_likelihoods)
        print i, sum_likelihoods
        print len(DISTANCE_CACHE.keys())

