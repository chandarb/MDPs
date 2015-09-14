import shapely
from shapely.geometry.point import Point
import numpy as np
from numpy import linalg
import scipy.stats as sps
import math
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import collections as mc

### Parameters to the model ###

# define the boundaries of the space
xmin = 0
xmax = 1
ymin = 0
ymax = 1

# the number of points to sample from
N = 100

# size of the terminal regions
c_size = .07
# the number of good terminal states
num_good = 2
# the number of bad terminal states
num_bad = 1
# total number of terminal states
total = num_good + num_bad

# the variance of each class in the mixture
# distribution that sampled points become associated with
sigma = .1
cov_s = np.identity(2) * sigma
# inverse covariance matrix
V_si = np.linalg.inv(cov_s)

# number of features associated with a point
d = 3
# mean of feature vector for each class
theta_mu = [[0,1,-1],[10,11,9], [-10,-9,-11]]

if len(theta_mu) != total:
    print "Each class should have one mean feature vector"
    raise

for i in theta_mu:
    if len(i) != d:
        print "The mean feature vectors should all have length " + str(d)
        raise

#theta_mu = [[0,1,-1]]

# covariance between the state and theta
Ap = np.ones((2,d)) * .1
A = np.transpose(Ap)
# covariance between the thetas
Vt = np.array([[1,.5,.5], [.5,1,.5], [.5,.5,1]])
for i in Vt:
    if len(i) != d:
        print "The feature covariance matrix should have trace" + str(d)
        raise

# full covariance matrix of theta and s
top = np.concatenate((cov_s, Ap), axis=1)
bottom = np.concatenate((A, Vt), axis=1)
cov_t = np.concatenate((top, bottom))

# conditional covariance
c_cov = Vt - np.dot(np.dot(A, V_si), Ap)

# scaling term in conditional mean
scale = np.dot(A, V_si)

# step size on each iteration
dist = .03
# noise in movement
delta = .0001
noise_mat = delta * np.identity(2)
# action sample fineness
fine = 30
# action noise sample fineness
s_fine = 5
# discount
gamma = .1
