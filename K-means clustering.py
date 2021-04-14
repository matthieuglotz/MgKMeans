"""
Title: 1. K-means clustering.py
Author: Matthieu Glotz
Started on: 11/04/2021

This python file creates a module a homemade for k-means clustering algorithm. A K means clustering algorithm does the
following:
1- It randomly picks K points in the dataset as group centroids (initialisation)
2- It allocates every point in the dataset to the nearest centroid (clustering)
3- it recalculates each group centroid.
The algorithm stops once the group centroids are stabilised (i.e do not change by more than an epsilon)
"""
import numpy as np
import pandas as pd


class MgKMeans(object):
    """
    Model object class for an implementation of the naive k means clustering algorithm.
    input: seed - numpy random seed generator (optional, defaults to 13)
    methods: fit - calculates K centroids based on a numpy array
             predict - predicts the cluster of observations in a numpy arrays after fitting
             get_centroids - get the value of the clusters centroids
    """

    def __init__(self, seed=13):
        self.random_seed = seed
        self.epsilon = 0.01
        self.centroids = []

    def pick_start_centroids(self, data, k):
        """
        input: k - number of clusters to generate
               data - numpy ndarray with the data to be clustered
        output: list of centroids randomly picked from the data
        """
        np.random.seed(self.random_seed)
        centroid_idx_arr = np.random.choice(data.shape[0], k)
        return [np.reshape(data[idx], (1, -1)) for idx in centroid_idx_arr]

    @staticmethod
    def calculate_point_distance(matrix_1, matrix_2=[]):
        """
        input: matrix_1 - a numpy ndarray of points
               matrix_2 (optional) - a second numpy ndarray of points. If unspecified, defaults to the 0 matrix,
               hence calculates matrix 1 points' norms.
        output: ndarray with the distance of each element of matrix 1 to the corresponding element of matrix 2.
                If matrix 2 is unspecified, then returns the norm of each element in matrix 1
        """
        if matrix_2 == []:
            matrix_2 = np.zeros(matrix_1.shape)

        return np.reshape(np.sum((matrix_1 - matrix_2) ** 2, axis=1) ** (1 / 2), (-1, 1))

    def calculate_centroid_distance(self, data, centroid):
        """
        input: centroid - numpy array point to calculate the distance to
               data - numpy ndarray with the data to be clusterised
        output: numpy array with each observation distance to the centroid
        """
        centroid_matrix = np.repeat(np.reshape(centroid, (-1, data.shape[1])), data.shape[0], axis=0)
        return self.calculate_point_distance(data, centroid_matrix)

    def get_centroids(self):
        """
        input: None
        output: returns centroids value
        """
        return self.centroids

    def predict(self, data):
        """
        input: data - numpy ndarray with the data to be clustered
        output: numpy array with the index of the nearest centroid for each observation in the data
        """
        if not self.get_centroids():
            raise Exception("Need to fit model before predicting clusters")
        distance_arr = np.concatenate([self.calculate_centroid_distance(data, centroid) for centroid in self.centroids],
                                      axis=1)
        return np.argmin(distance_arr, axis=1)

    def convergence_check(self, previous_centroids):
        """
        input: Previous centroid values
        output: Boolean value indicating whether the centroids have converged
        """
        centroid_matrix = np.concatenate(self.centroids, axis=0)
        previous_centroid_matrix = np.concatenate(previous_centroids, axis=0)
        distance_matrix = self.calculate_point_distance(centroid_matrix, previous_centroid_matrix)
        norm_matrix = self.calculate_point_distance(previous_centroid_matrix)
        return np.all(distance_matrix / norm_matrix < self.epsilon)

    @staticmethod
    def update_centroids(clusters, data):
        """
        input: data - numpy ndarray with the data to be clustered
               clusters - cluster index of each point in the data
        output: new centroid list based on the points value in the cluster
        """
        centroids_df = pd.DataFrame(np.concatenate([clusters, data], axis=1)).groupby(0).mean()
        return [centroids_df.loc[idx].to_numpy() for idx in centroids_df.index]

    def fit(self, k, data):
        """
        input: k - number of clusters to generate
               data - numpy ndarray with the data to be clusterised
        output: None (updates centroids until convergence)
        """

        # Initialisation
        self.centroids = self.pick_start_centroids(data, k)
        previous_centroids = [np.zeros((1, data.shape[1])) for dim in range(k)]

        # Iterations
        while self.convergence_check(previous_centroids) is False:
            previous_centroids = self.get_centroids()
            clusters = self.predict(data)
            self.centroids = update_centroids(clusters, data)
