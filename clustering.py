# clustering.py
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def find_optimal_k(X_lsa):
    X_cosine_distance = 1 - pairwise_distances(X_lsa, metric='cosine')
    inertias = []
    K_values = range(1, 100)

    for k in K_values:
        kmeans_model = KMeans(n_clusters=k, n_init=1)
        kmeans_model.fit(X_cosine_distance)
        inertias.append(kmeans_model.inertia_)

    upper_bound = 50
    inertias = [i / X_cosine_distance.shape[0] for i in inertias]

    plt.plot(K_values[:upper_bound], inertias[:upper_bound], 'o-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

    return inertias, K_values

def show_kmeans(inertias, K_values):
    upper_bound = 50
    plt.plot(K_values[:upper_bound], inertias[:upper_bound], 'o-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()