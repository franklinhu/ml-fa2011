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
import time

import cross_validation

NUM_ARGS = 3
NUM_FOLDS = 5
CURRENT_FOLD = None
WIDTH = 19

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

# Cache the following since computations are expensive
KTH_NEAREST_CACHE = collections.OrderedDict()

## Calculates the great circle distance between two point on the earth's
## surface in degrees. loc1 and loc2 are pairs of longitude and latitude. E.g.
## int(dist_deg((10,0), (20, 0))) gives 10
def dist(loc1, loc2):
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    tmp = math.sin(lat1 * DEG2RAD) * math.sin(lat2 * DEG2RAD) \
                  + math.cos(lat1 * DEG2RAD) * math.cos(lat2 * DEG2RAD)\
                  * math.cos((lon2 - lon1) * DEG2RAD)
    # Floating point rounding issues for tmp = 1.0000000000...1
    # cause domain error when taking arccos, so we clamp it between -1,1
    tmp = min(max(tmp, -1), 1)
    return math.acos(tmp) * RAD2DEG

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

def kernel_density_base(distance, width, data, query_point, k, N, fold_num):
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
    total = 0
    for x_i in data(fold_num):
        w = width(k, data, x_i, query_point)
        total += (math.e ** (-distance(query_point, x_i) / w)) / w
        #total += distance(query_point, x_i) / w + math.log(w)
    #return -total / N
    return math.log(total) - math.log(N)

def log_kernel_density_a(data, query_point, k, N, fold_num):
    """
    Kernel density function as given by:
    P(x) = (1/N) * \sum\limits_{i=1}^N K_b(d(x,x_i))
    """
    return kernel_density_base(dist, lambda k,data,x_i,x: WIDTH, 
                               data, query_point, k, N, fold_num)

def log_kernel_density_b(data, query_point, k, N, fold_num):
    """
    P(x) = k / (2N * d_k(x))
    where d_k(x) is the distance from x to the kth nearest neighbor of x
    """
    return math.log(k) - math.log(2 * N * dist_k(k, data, query_point))

def log_kernel_density_c(data, query_point, k, N, fold_num):
    """
    P(x) = (1/N) * \sum\limits_{i=1}^N K_{d_k(x)}(d(x,x_i))
    same as (a) except the kernel width is determined by the kth
    nearest neighbor distance
    """
    return kernel_density_base(dist,
                               lambda k,data,x_i,x: dist_k(k, data, x),
                               data, query_point, k, N, fold_num)

def log_kernel_density_d(data, query_point, k, N, fold_num):
    """
    P(x) = (1/N) * \sum\limits_{i=1}^N K_{d_{ik}(x)}(d(x,x_i))
    same as (c) except the kernel width is the kth nearest neighbor from 
    each x_i to its kth nearest neighbor, rather than from x

    Since this kernel width doesn't depend on the query_point, it can be
    cached and then used for each query_point in the validation set
    """
    total = 0
    for x_i in data(fold_num):
        key = (x_i, CURRENT_FOLD)
        if key not in KTH_NEAREST_CACHE:
            KTH_NEAREST_CACHE[key] = dist_k(k, data, x_i)
        w = KTH_NEAREST_CACHE[key]
        total += dist(query_point, x_i) / w + math.log(w)
    return -total / N

DENSITIES = {
    "a": log_kernel_density_a,
    "b": log_kernel_density_b,
    "c": log_kernel_density_c,
    "d": log_kernel_density_d
}

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "USAGE: $ python density_estimation.py data_file " \
              "kernel_density k"
        print "\tdata_file      -- input csv file"
        print "\tkernel_density -- kernel density to use (a,b,c,d)"
        print "\tk              -- k value for k-NN"
        sys.exit(1)

    data_file = sys.argv[1]
    if sys.argv[2] in DENSITIES:
        kernel_density = DENSITIES[sys.argv[2]]
    else:
        raise Exception('No such kernel density. Choices: (a,b,c,d)')

    k = int(sys.argv[3])

    handle = open(data_file, 'r')
    handle.readline()
    csv_file = csv.reader(handle)
    data = []

    LON = -73.125
    DELTA = 15
    for line in csv_file:
        l = tuple(map(lambda x: float(x), line[0:2]))
        if abs(l[0] - LON) < DELTA:
            data.append(l)

    print len(data)

    # Find optimal width for (a)
    if kernel_density == log_kernel_density_a:
        try:
            print "Finding optimal width for (a)"
            widths = range(15, 35)
            NUM_FOLDS = 20
            cv = cross_validation.CrossValidation(NUM_FOLDS, data, True)
            for i in xrange(NUM_FOLDS):
                WIDTH = widths[i]
                CURRENT_FOLD = i
                sum_likelihood = 0
                t = time.time()
                for v_ex in cv.validation_examples(i):
                    N = cv.num_training_examples(i)
                    log_likelihood = kernel_density(cv.training_examples, 
                                                    v_ex, 1, N, CURRENT_FOLD)
                    sum_likelihood += log_likelihood
                print "Fold %d, width: %d, likelihood: %f, time: %f" % (i, 
                    WIDTH, sum_likelihood, time.time() - t)
        except KeyboardInterrupt:
            print "Aborting optimal width for (a)"
            print "Continuing"

    print "Beginning cross validation of likelihoods"
    print k
    NUM_FOLDS = 5
    random.shuffle(data)
    data = data[:7000]
    cv = cross_validation.CrossValidation(NUM_FOLDS, data, True)
    for i in xrange(NUM_FOLDS):
        CURRENT_FOLD = i
        print "Current fold: %d" % i
        sum_likelihood = 0 
        t = time.time()
        for v_ex in cv.validation_examples(i):
            N = cv.num_training_examples(i)
            log_likelihood = kernel_density(cv.training_examples,
                                          v_ex, k, N, CURRENT_FOLD)

            sum_likelihood += log_likelihood
            
        print "Fold %d, likelihood: %f, time: %f" % (i, sum_likelihood,
                time.time() - t)

