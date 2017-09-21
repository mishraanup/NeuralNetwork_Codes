"""
Author: Anup Mishra
Code: Competitive Learning - SOM
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

# Sigma for gaussian function
sigma0 = 0.3

# Learning Rate
eta0 = 0.00007

# Time Constant
T = 700

# NN Epochs
max_epochs = 1000

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
def cluster_som(n, m, d):

    # seed random numbers to make calculation deterministic
    # (just a good practice)
    np.random.seed(1)
    global eta0, sigma0

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

    # This would contain the distances of the winning op neuron to other neurons
    distance_mat = np.zeros(m)

    # Neighbourhood function matrix
    distance_mat_nh = np.zeros(m)

    # Initial values of eta and sigma are considered
    eta = eta0
    sigma = sigma0

    for i in range(max_epochs):
        print("Epoch ->", i)
        for j in range(len(d)):

            # Competition
            l1 = nonlin(np.dot(d[j, :], w0))  # l1 is the output layer
            # print(l1)
            o = np.argmax(l1)
            cluster[j] = o
            # print("o:", o)

            # Cooperation
            # This will first have to find the distances of the current op neuron with rest of the neurons
            # 1D lattice Considered
            for z in range(0, m, 1):
                # print("for z: ", z)
                distance_mat[z] = abs(o - z)
                distance_mat_nh[z] = np.exp(-(distance_mat[z]*distance_mat[z]/(2*sigma*sigma)))
            # print("Distances", distance_mat)
            # print("Neighbourhood Function: ", distance_mat_nh)

            # Introduction of Forgetting Term
            # do this for each op neuron
            for z in range(0, m, 1):
                diff = (d[j, :] - w0[:, z])
                diff = diff * distance_mat_nh[z]
                w0[:, z] += eta * diff

        # Updating the Learning rate
        eta = eta0 * np.exp(-(i/T))
        sigma = sigma0 * np.exp(-(i/T))

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
            # print(cluster_centers_previous)

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
        clusters = 6  # clusters
        data = get_data(atts)
        # print data
        print("Number of data points: ", len(data), "\n")
        final_weights = cluster_som(atts, clusters, data)
        # Final Weights
        print("\n Final Weights: \n", final_weights)

    except OSError as e:
        print("Error", e, "\n")
