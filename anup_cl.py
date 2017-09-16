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
eta = 0.00001

# NN Epochs
max_epochs = 2269

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
# d = data set without labels
def cluster_cl(n, m, d):

    # seed random numbers to make calculation deterministic
    # (just a good practice)
    np.random.seed(1)
    global eta

    # initialize weights randomly
    w0 = np.random.random((n, m))
    # print("Initial weights: \n", w0)

    # cluster vector
    cluster = np.zeros(len(d))

    # cluster Centers:
    cluster_centers = np.zeros((m, n))
    cluster_centers_previous = np.zeros((m, n))

    # cluster center distance array
    cluster_center_dist_matrix_array = np.zeros(max_epochs)

    for i in range(max_epochs):
        print("Epoch -->", i)
        for j in range(len(d)):
            l1 = nonlin(np.dot(d[j, :], w0))  # l1 is the output layer
            o = np.argmax(l1)
            cluster[j] = o
            # print(o)
            diff = (d[j, :] - w0[:, o])
            # print eta * diff
            w0[:, o] += eta * diff

        # Updating the Learning rate
        # eta = 0.001 * .8

        # Get cluster centers
        for k in range(0, m, 1):  # go over each cluster data
            index_list = np.where(cluster == k)
            for l in range(0, n, 1):  # go over each attribute
                cluster_centers[k, l] = np.sum(d[index_list[0], l]) / len(index_list[0])
        """
        STOPPING CRITERIA 1: CLUSTERS SHOULD BE WELL SEPARATED
        """
        # Cluster Center Distances
        cluster_center_dist_matrix = distance.pdist(cluster_centers, metric='euclidean')

        # Print if the cluster centers look good with the number of epochs
        if not np.isnan(cluster_center_dist_matrix.mean()):
            cluster_center_dist_matrix_array[i] = cluster_center_dist_matrix.mean()
        else:
            cluster_center_dist_matrix_array[i] = cluster_center_dist_matrix_array[i-1]

        """
        STOPPING CRITERIA 2: CLUSTER CENTERS NOT UPDATING
        """
        # Cluster Center Changes
        if i == 1:
            cluster_centers_previous = cluster_centers
        else:
            cluster_centers_previous = cluster_centers - cluster_centers_previous
            print(cluster_centers_previous)

    # print(cluster_center_dist_matrix_array)
    ideal_max_epochs = np.argmax(cluster_center_dist_matrix_array)

    if (max_epochs - ideal_max_epochs)-1 == 0:
        print("\n \n VERY GOOD. You have used the optimal number of epochs for maximum cluster separation!!")
    else:
        print("\n \n NOTE: Ideal Max Epoch Number based on cluster center distance is  : ",
              ideal_max_epochs, "\n PLEASE UPDATE 'max_epochs'!")

    # Show Clusters
    fig = pl.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(cluster_center_dist_matrix_array)

    fig = pl.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(d[:, 0], d[:, 1], d[:, 3], c=cluster)
    print("\n Final Cluster Centers: \n", cluster_centers)
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 3], c='r', s=100)
    pl.show()
    return w0


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
        return d_norm

    except ImportError as error:
        print("Error", error, "\n")


if __name__ == "__main__":
    try:
        atts = 4  # attributes
        clusters = 3  # clusters
        data = get_data(atts)
        # print data
        print("Number of data points: ", len(data), "\n")
        final_weights = cluster_cl(atts, clusters, data)
        # Final Weights
        print("\n Final Weights: \n", final_weights)

    except OSError as e:
        print("Error", e, "\n")
