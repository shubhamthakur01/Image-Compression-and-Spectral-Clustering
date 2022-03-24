import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')
import cv2


def kmeans_plus_centroids(X: np.ndarray,  k: int)-> (np.array):
    '''
    :param X: input array
    :param k: cluster size
    :return: Initialize the  centroid based on kmeans++ algorithm.
    '''
    index = np.random.choice(X.shape[0], 1, replace=False)
    centroids = []
    centroids.append(X[index][0])
    for cluster in range(k - 1):
        centroid_index = np.argmax(np.min(distance.cdist(centroids, X, 'euclidean'), axis=0))
        centroids.append(X[[centroid_index]][0])
    return np.array(centroids)


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2)-> (np.ndarray, np.array):
    '''

    :param X: input array
    :param k: cluster size
    :param centroids: cluster intialization method, Could be kmeans++, None or custom initialized
    :param max_iter: number of iteration before cluster stops updating
    :param tolerance: difference in the cluster centers of two consecutive iterations to declare convergence.
    :return: centroid : final evaluated centroid coordinated based on max_iter and tolerance
             labels : final cluster labels for each data points in X
    '''
    if centroids == 'kmeans++':
        centroids = kmeans_plus_centroids(X, k)
    elif centroids is None:
        index = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[index]

    norm = tolerance + 1
    distances = distance.cdist(centroids, X, 'euclidean')
    labels = np.argmin(distances, axis=0)
    iter_ = 0
    while iter_ < max_iter and norm > tolerance:
        distances = distance.cdist(centroids, X, 'euclidean')
        labels = np.argmin(distances, axis=0)
        norm = 0
        for i in range(k):
            X_i = X[labels == i]
            if len(X_i) == 0:
                index_ = np.random.choice(X.shape[0], 1, replace=False)
                centroid_i = X[index_][0]
            else:
                centroid_i = np.mean(X_i, axis=0)
            norm += np.sum(np.square(centroid_i - centroids[i]))
            centroids[i] = centroid_i
        norm = np.sqrt(norm) / k
        iter_ += 1
    return centroids, labels


'''These below functions are helper functions used for computing Affinity matrix using Breiman's RF'''
def conjure_twoclass(X : pd.DataFrame)-> (pd.DataFrame, pd.Series):
    '''

    :param X: Input DataFrame
    :return: X_synth: Returns synthetically generated data with labels
    '''
    X_rand = df_scramble(X)
    X_synth = pd.concat([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X),dtype=int),np.ones(len(X_rand),dtype=int)],axis=0)
    return X_synth.to_numpy(), pd.Series(y_synth)


def df_scramble(X : pd.DataFrame) -> pd.DataFrame:
    '''
    :param X: Input DataFrame
    :return: shuffled version of dataframe X
    '''
    X_rand = X.copy()
    X_rand = X_rand.sample(frac=1).reset_index(drop=True)
    return X_rand


def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples


def get_similarity(leaf_samples,n_estimators):
    '''
    :param leaf_samples: list of arrays where each array is the set of X sample indexes residing in a single leaf of some tree in rf forest.
    :param n_estimators: number of trees
    :return: similarity matrix for
    '''
    similarity  = np.zeros((500,500))
    for leaf in leaf_samples:
        for i in range(len(leaf)):
            for j in range(0, len(leaf)):
                similarity[leaf[i], leaf[j]] += 1
    similarity /= n_estimators
    return similarity


def get_labels(X:np.ndarray, n_estimators=500):
    '''
    :param X: Input data
    :param n_estimators: number of trees
    :return: labels for all X and similarity matrix.
    '''
    df = pd.DataFrame(X,columns=['col_1', 'col_2'])
    X_synth,y_synth = conjure_twoclass(df)
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=50)
    rf.fit(X_synth,y_synth)
    leaf_sample = leaf_samples(rf, X)
    similarity = get_similarity(leaf_sample,n_estimators)
    cluster = SpectralClustering(n_clusters=2)
    labels = cluster.fit_predict(similarity)
    return labels, similarity