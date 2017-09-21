"""
Author: Anup Mishra
Code: Competitive Learning
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import pylab as pl
import warnings
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

# Fix for warning messages
warnings.filterwarnings("ignore")

# globals

# Data set location:
data_name = "iris.data"

# Learning Rate
eta0 = 0.1

# NN Epochs
max_epochs = 1000000

# Time Constant
T = 1000000

# Criterion
CLUSTER_DISTANCE = 1  # DONE
CLUSTER_CENTER_CHANGE = 2  # DONE
CLUSTER_VALIDITY = 3  # TODO


# sigmoid function
def nonlin(x, deriv=False):
    if deriv is True:
        return x*(1-x)
    return 1/(1+np.exp(-x))


# Function to compute clusters
# INPUTS:
# n = attributes
# m = clusters
# h = hidden layer nodes
# d = data set without labels
# y = hot key label matrix
def cluster_bp(n, m, h, d, y):

    # seed random numbers to make calculation deterministic
    # (just a good practice)
    np.random.seed(1)
    global eta0

    # initialize weights randomly
    w0 = 2 * np.random.random((n, h)) - 1
    w1 = 2 * np.random.random((h, m)) - 1

    for i in range(max_epochs):
        l0 = d
        l1 = nonlin(np.dot(l0, w0))  # l1 is the hidden layer
        l2 = nonlin(np.dot(l1, w1))  # l2 is the output layer

        # This is the error at the output neuron
        l2_error = y-l2

        if (i % 10000) == 0:
            print("Epoch: ", i, "Error:" + str(np.mean(np.abs(l2_error))))

        l2_del_w = l2_error*nonlin(l2, deriv=True)

        l1_error = l2_del_w.dot(w1.T)

        l1_del_w = l1_error * nonlin(l1, deriv=True)

        eta = eta0 * np.exp(-(i / T))

        w0 += eta * l0.T.dot(l1_del_w)
        w1 += eta * l1.T.dot(l2_del_w)

    # Show Clusters
    return w0, w1


def get_data(dimensions):
    try:
        d = pd.read_csv('iris.data', header=None)
        d = d.ix[:, 0:dimensions-1]

        d = d.values
        # print("Data: \n", d)
        d_norm = d
        for i in range(0, dimensions, 1):
            normalization_factor = np.linalg.norm(d[:, i])
            d_norm[:, i] = d[:, i] / normalization_factor
        # Get output Data for IRIS
        op = np.concatenate((np.tile((1, 0, 0), (50, 1)),
                             np.tile((0, 1, 0), (50, 1)),
                             np.tile((0, 0, 1), (50, 1))),
                            axis=0)
        return d_norm, op

    except ImportError as error:
        print("Error", error, "\n")


if __name__ == "__main__":
    try:
        atts = 4  # attributes
        clusters = 3  # clusters
        hids = 3  # Hidden layer nodes
        data, labels = get_data(atts)
        # print data
        print("Number of data points: ", len(data), "\n")
        final_weights0, final_weights1 = cluster_bp(atts, clusters, hids, data, labels)
        # Final Weights
        print("\n Final Weights: \n", final_weights0, "\n", final_weights1)

    except OSError as e:
        print("Error", e, "\n")
