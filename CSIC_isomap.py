import torch
import numpy as np
import sklearn
import sklearn.manifold

"""
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def knn_matrix(X, n_neighbors=20):"""
# test
"""
    Returns a k-nearest neighbor matrix for the given matrix X.
    :param X: A torch tensor of shape (n, d) where n is the number of samples and d is the dimension of the samples.
    :param n_neighbors: The number of neighbors to consider.
    :return: A torch tensor of shape (n, n) where the (i, j)th entry is the distance between the ith and jth samples.
    """
# test
"""n = X.shape[0]
    X = X.to(default_device)
    distances = torch.cdist(X, X)
    _, indices = torch.topk(distances, n_neighbors + 1, largest=False)
    indices = indices[:, 1:]
    distances = torch.gather(distances, 1, indices)
    return distances


def isomap():
"""

