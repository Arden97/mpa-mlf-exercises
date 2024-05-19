import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from numpy.linalg import norm
from matplotlib.image import imread
from sklearn.cluster import KMeans

loaded_points = np.load('C:/Users/Artem/Study/VUT_FEKT/MLF/mpa-mlf-exercises/Lab_02/Data/k_mean_points.npy')
plt.figure(1)
plt.scatter(loaded_points[:,0],loaded_points[:,1])
k = 3

def initialize_clusters(points: np.array, k_clusters: int) -> np.array:
    """
    Initializes and returns k random centroids from the given dataset.

    :param points: Array of data points.
    :type: points ndarray with shape (n, 2)
    
    :param k_clusters: The number of clusters to form
    :type k_clusters: int 


    :return: initial_clusters
    initial_clusters: Array of initialized centroids

    :rtype:
    initial_clusters: np.array (k_clusters, 2)
    :
    
    """
    
    ###################################
    # Write your own code here #
    # Random shuffle
    vector_with_all_indexes = np.arange(points.shape[0])
    vector_with_all_indexes = np.random.permutation(vector_with_all_indexes)
    required_indexes = vector_with_all_indexes[:k_clusters]
    return points[required_indexes]

    # Different method
    # number_of_points = len(points)
    # numbers = random.choice(number_of_points, size = 3, replace=False)
    # return initial_clusters[numbers]

def calculate_metric(points: np.array, centroid: np.array) -> np.array:
    """
    Calculates the distance metric between each point and a given centroid.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)
    
    :param centroid: A single centroid
    :type centroid: ndarray with shape (1, 2)

    :return: distances_array
    distances_array: Array of distances from point to centroid

    :rtype:
    distances_array: ndarray with shape (n,)
    :
    """

    ###################################
    # Write your own code here #

    return np.square(norm(points-centroid, axis=1))

    ###################################

def compute_distances(points: np.array, centroids_points: np.array) -> np.array:
    """
    Computes and returns the distance from each point to each centroid.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)

    :param centroids_points: A all centroid points
    :type centroids_points: ndarray with shape (k_clusters, 2)
    

    :return: distances_array
    distances_array: 2D array with distances of each point to each centroid.

    :rtype:
    distances_array: ndarray of shape (k_clusters, n)
    """
    ###################################
    # Write your own code here #

    return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])

    ###################################

def assign_centroids(distances: np.array) -> np.array:
    """
    Assigns each point to the closest centroid based on the distances.

    Parameters:
    :param distances: 2D array with distances of each point to each centroid.
    :type distances: ndarray with shape (k_clusters, n)

    :return: assigned_clusters
    assigned_clusters: Array indicating the closest centroid for each data point.

    :rtype:
    assigned_centroids: ndarray with shape (1, n) and dtype = np.int32
    """

    ###################################
    # Write your own code here #
    return np.argmin(distances, axis = 1)

    ###################################

def calculate_objective(assigned_centroids: np.array, distances: np.array) -> np.array:
    """
    Calculates and returns the objective function value for the clustering.

    Parameters:
    :param assigned_centroids: Array indicating the cluster assignment for each point.
    :type assigned_centroids: ndarray with shape (1, n) and and dtype = np.int64
    
    :param distances: 2D array with distances of each point to each centroid
    :type distances: ndarray with shape (k_clusters, n) and and dtype = np.float64

    :return: onjective_function_value
    onjective_function_value: Objective function value.

    :rtype:
    onjective_function_value: float32

    
    """
    ###################################
    # Write your own code here #

    distances = distances.T
    # calculates how far away each point in a cluster is from the center dot of that cluster
    selected_min =  distances[np.arange(len(distances)), assigned_centroids]

    ###################################
    
    return np.sum(selected_min)

def calculate_new_centroids(points: np.array, assigned_centroids: np.array, k_clusters: int) -> np.array:
    """
    Computes new centroids based on the current cluster assignments.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)

    :param assigned_centroids: Array indicating the closest centroid for each data point.
    :type assigned_centroids: ndarray with shape (1, n) and dtype = np.int32
    

    :param k_clusters: Number of clusters.
    :type k_clusters: int


    :return: new_clusters
    new_clusters: new cluster points

    :rtype:
    new_clusters: ndarray with shape (1, n) and dtype = np.float32
    """
    
    ###################################
    # Write your own code here #

    new_clusters = []
    for cluster_id in range(k_clusters):
        i = np.where(assigned_centroids == cluster_id)
        points_in_cluster = points[i]
        new_clusters.append(np.mean(points_in_cluster, axis=0))

    ###################################
    
    return np.array(new_clusters)

def fit(points: np.array, k_clusters: int, n_of_oterations: int, error: float = 0.001) -> tuple:
    """
    Fits the k-means clustering model on the dataset.

    Parameters:
    :param points : Array of data points.
    :type points: ndarray with shape (n, 2) and dtype = np.float32

    :param k_clusters:  Number of clusters
    :type k_clusters: int

    :param n_of_oterations:  Maximum number of iterations
    :type n_of_oterations: int

    
    :param error: Threshold for convergence.
    :type error: float

    :return: centroid_points, last_objective
    centroid_points: final centroid points
    last_objective: final objective funtion

    :rtype:
    centroid_points: ndarray with shape (k_clusters, 2) and dtype = np.float32
    last_objective: float
    
    """

    ###################################
    # Write your own code here #
    
    centroid_points = initialize_clusters(points, k_clusters)
    last_objective = 10000
    for _ in range(n_of_oterations):
        distances = compute_distances(points, centroid_points)
        cluster_belongs = np.argmin(distances, axis=0)
        objective = calculate_objective(cluster_belongs, distances)
        if abs(last_objective - objective) < error:
            break
        last_objective = objective
        centroid_points = calculate_new_centroids(points, cluster_belongs, k_clusters)

    ###################################
    
    return centroid_points, last_objective

n_of_iterations = 100
centroids, objective = fit(loaded_points, k, n_of_iterations)
plt.figure(2)
plt.scatter(loaded_points[:,0],loaded_points[:,1])
plt.scatter(centroids[:,0].T,centroids[:,1].T)


k_all = range(2, 10)
all_objective = []
for n_of_cluster in k_all:
    centroids, objective = fit(loaded_points, n_of_cluster, n_of_iterations)
    all_objective.append(objective)
plt.figure(3)
plt.plot(k_all, all_objective)
plt.xlabel('K clusters')
plt.ylabel('Sum of squared distance')

loaded_image = imread('C:/Users/Artem/Study/VUT_FEKT/MLF/mpa-mlf-exercises/Lab_02/Data/fish.jpg')
plt.figure(3)
plt.imshow(loaded_image)

plt.show()



