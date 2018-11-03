import numpy as np
import matplotlib.pyplot as plt
import math
import sys

m =  24         # Number of features
n = 32000       # Number of training data points
alpha = 0.001    # learning rate
lmbda = 0   # regularisation parameter
# thresh = 0.5
p = 2           # p-norm will be used
nIterations = 20000
nHiddenUnits = 100
batchSize = 1
nOutputUnits = 3
################################################################################

train_file = "train.csv" #sys.argv[1]
test_file = "test.csv" #sys.argv[2]
out_file = 'submission.csv'


day2num = {                                 # Dictionary to map days to numbers
    'monday': 1, 'tuesday': 2,
    'wednesday': 3, 'thursday': 4,
    'friday': 5, 'saturday': 6,
    'sunday': 7
}

def load_data():
    # Loads data from the file
    # adds padding of 1s to the left and top
    # Slices the resultant matrix in X(features) and y(target)
    X = np.genfromtxt(train_file, delimiter=",", dtype=str, encoding='utf8',  unpack=True)

    for x in X:
        x[0] = 1                    # Repalcing titles with 1
    i = 1

    while i < len(X[22]):           # Converting dates to numbers
        X[22][i] = day2num[X[22][i]]
        X[23][i] = day2num[X[23][i]]
        i += 1
    x_0 = np.ones_like(X[0])
    X = np.concatenate(([x_0], X), axis = 0)
    X = X.astype(float)

    X = X.transpose()
    X = X[1:]
    np.random.shuffle(X)
    X = X.transpose()
    y = X[-1]
    X = X[0:-1]
    
    return X, y


def normalisation(x):
    # Does feature scaling using the formula :
    #                         x - min(x)
    #                    ------------------
    #                     max(x)  - min(x)

    avg = np.average(x)
    diff = np.max(x) -  np.min(x)

    x = ( x - np.min( x ) ) / diff
    return x, avg, diff, np.min(x), np.std(x)
