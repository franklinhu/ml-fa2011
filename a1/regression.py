from numpy import *
from numpy import matrix
from numpy import linalg
import numpy as np
import csv

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
   pass

def localLinearRegressionForP1(x, data):
   pass

def localLinearRegressionForP2(x, data):
   pass

def localLinearRegressionForS1(x, data):
   pass

def localLinearRegressionForS2(x, data):
   pass


## Estimate the residual time using locally weighted
## regression with Gaussian or Laplacian kernel
## Outputs estimate(float)
def localWeightedRegression(station, phase, x, data):
   pass

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


