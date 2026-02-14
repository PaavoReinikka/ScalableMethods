import numpy as np
from numba import njit
import pandas as pd
from scipy.cluster.vq import vq
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from scipy.stats import mode
import matplotlib.pyplot as plt


def laplacian(S, normalize='rw'):
    """
        S: similarity matrix
        normalize: 'rw' for random walk, 'sym' for symmetric, None for unnormalized
    """
    if(normalize=='rw'):
        return np.diag(1./np.sum(S,axis=1)) @ (np.diag(np.sum(S,axis=1)) - S)
    elif(normalize=='sym'):
        return np.sqrt(np.diag(1./np.sum(S,axis=1))) @ (np.diag(np.sum(S,axis=1)) - S) @\
        np.sqrt(np.diag(1./np.sum(S,axis=1)))
    else:
        return np.diag(np.sum(S,axis=1)) - S

    
def laplacian2embedding(L, n_components=1, normalized=False):
    """
        L: Laplacian matrix
        n_components: number of eigenvectors to return
        normalized: whether to normalize the returned eigenvectors 
    """
    #Assumes a fully connected graph (i.e., drops the first [minor] eigen vec by default)
    evals, evecs = np.linalg.eig(L)
    index = sorted(range(evals.shape[0]), key=lambda k: evals[k] )
    res = evecs[:,index[1:n_components+1]]
    if(normalized):
        return normalize(res, axis=1)
    return res



def kmeans_objective(D, labels):
    obj = 0
    for label in np.unique(labels):
        D2 = D[np.where(labels==label)]**2
        obj += np.sum(D2)
    return obj/D.shape[0]


def matching_dist(X1, X2, aggregator = np.mean):
    n1, n2 = X1.shape[0], X2.shape[0]
    result = np.empty((n1, n2))
    for i in range(n1):
        for j in range(n2):
            result[i,j]=aggregator(X1[i,:]!=X2[j,:])
            
    return result

def euclidean_dist(X1, X2):
    #TODO-alternative: implement pairwise distance function(s) wich return
    # a matrix of shape (n, k) for inputs with following sizes: 
    #  X1.shape=(n,d), X2.shape=(k,d)
    return distance_matrix(X1, X2, p=2)
    
    
def manhattan_dist(X1, X2):
    #TODO-alternative: implement pairwise distance function(s) wich return
    # a matrix of shape (n, k) for inputs with following sizes: 
    #  X1.shape=(n,d), X2.shape=(k,d)
    return distance_matrix(X1, X2, p=1)
    

def assign(D, k):#metric):
    '''Takes the distances to k current cluster representatives (last ,
       and returns the cluster assignments and the distance to
       the closest representative (for each datapoint)
    '''
    ind = np.argmin(D, axis=1)
    bool_ind = (np.arange(k).reshape(-1,1)==ind).T
    
    return ind, D[bool_ind]

def update_representatives(X, assignments, representatives, method):
    '''Based on provided assignments, iterates through every cluster
       and calculates the new representatives using method (e.g., np.mean/median or scipy.stats.mode).
    '''
    for i in np.unique(assignments):
        subX = X[np.where(assignments==i),:][0]
        representatives[i] = method(subX, axis=0)
    
    return representatives


def KRepresentatives(X, k, x_init, n_iters = 100, dist_func = euclidean_dist, rep_func=np.mean):
    
    representatives = x_init
    
    #scores_arr = np.empty((n_iters,1))
    #representatives_arr = np.empty((k,n_iters, X.shape[1]))
    
    for i in range(n_iters):
        D = dist_func(X, representatives)
        assignments, dist = assign(D,k)
        representatives = update_representatives(X, assignments, representatives, rep_func)
        
        #these are optionally used to get the whole optimization path
        #score_arr[i] = score(dist)
        #representatives_arr[:,i,:] = representatives
        
    return assignments#, representatives_arr, scores_arr
    
    

def my_mode(x, axis):
    #scipy's mode returns a tupple, and therefore needs a wrapper
    return mode(x, axis=axis)[0]

### Spherical KMeans

@njit(cache=True)
def np_all_axis0(x):
    """Numba compatible version of np.all(x, axis=0)."""
    out = np.ones(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_and(out, x[i, :])
    return out

@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@njit(cache=True)
def np_any_axis0(x):
    """Numba compatible version of np.any(x, axis=0)."""
    out = np.zeros(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out

@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out

# TODO: add a function to replace un-used centroids, e.g., by random re-initialization

@njit
def kmeans(X, k, n_iter, init_centroids):
    #Fast parallel kmeans
    #Original implementation from old numba examples (here slightly modified)
    N = X.shape[0]
    D = X.shape[1]
    centroids = init_centroids

    for l in range(n_iter):
        dist = np.array([[np.sqrt(np.sum((X[i, :] - centroids[j, :])**2))
                          for j in range(k)] for i in range(N)])

        predictions = np.array([dist[i, :].argmin() for i in range(N)])

        centroids = np.array([[np.sum(X[predictions == i, j])/np.sum(predictions == i)
                               for j in range(D)] for i in range(k)])

    return centroids, dist, predictions


@njit
def kmeans_spherical_v1(X, k, n_iter, init_centroids):

    # This implementation collapses empty clusters to origin.
    # The origin tends to pull other clusters towards it.
    # Not good for high values of k.
    
    N = X.shape[0]
    D = X.shape[1]
    centroids = init_centroids

    for i in range(centroids.shape[0]):
        centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    for l in range(n_iter):
        dist = 1 - X@centroids.T

        predictions = np.array([dist[i, :].argmin() for i in range(N)])
        
        centroids = np.array([[np.sum(X[predictions == i, j])/max(1,np.sum(predictions == i))
                               for j in range(D)] for i in range(k)])

        for i in range(centroids.shape[0]):
            centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    return centroids, dist, predictions

@njit
def kmeans_spherical_v2(X, k, n_iter, init_centroids):
    
    # This implementation drops empty clusters

    N = X.shape[0]
    D = X.shape[1]
    centroids = init_centroids

    for i in range(centroids.shape[0]):
        centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])
        
    for l in range(n_iter):
        dist = 1 - X@centroids.T

        predictions = np.array([dist[i, :].argmin() for i in range(N)])

        centroids = np.array([[np.sum(X[predictions == i, j])/np.sum(predictions == i)
                               for j in range(D)] for i in np.unique(predictions)])
        
        for i in range(centroids.shape[0]):
            centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])
            
    return centroids, dist, predictions

@njit
def kmeans_spherical_v3(X, k, n_iter, init_centroids):
    
    # This keeps empty clusters and does not pull other clusters towards the origin.
    # It is the same as v2 but with a different way of handling empty clusters.
    # You can also change the way empty clusters are handled -- e.g., by random re-initialization.
    
    N = X.shape[0]
    D = X.shape[1]
    centroids = init_centroids

    for i in range(centroids.shape[0]):
        centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    for l in range(n_iter):
        dist = 1 - X@centroids.T

        predictions = np.array([dist[i, :].argmin() for i in range(N)])

        centroids_new = np.array([[np.sum(X[predictions == i, j])/max(1,np.sum(predictions == i))
                               for j in range(D)] for i in range(k)])
        
        mask = np_all_axis1(centroids_new == 0)
        centroids_new[mask,:] = centroids[mask,:]
        
        centroids = centroids_new
        
        for i in range(centroids.shape[0]):
            centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    return centroids, dist, predictions

@njit
def kmeans_spherical_v4(X, k, n_iter, init_centroids, discard_freq=10):

    # This keeps empty clusters and does not pull other clusters towards the origin.
    # It is the same as v2 but with a different way of handling empty clusters.
    # You can also change the way empty clusters are handled -- e.g., by random re-initialization.

    N = X.shape[0]
    D = X.shape[1]
    centroids = init_centroids

    for i in range(centroids.shape[0]):
        centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    discarded_centroids = np.zeros((k,1))
    for l in range(n_iter):
        dist = 1 - X@centroids.T

        predictions = np.array([dist[i, :].argmin() for i in range(N)])

        centroids_new = np.array([[np.sum(X[predictions == i, j])/max(1,np.sum(predictions == i))
                                   for j in range(D)] for i in range(k)])

        # TODO: add a function to replace un-used centroids, e.g., by random re-initialization
        mask = np_all_axis1(centroids_new == 0)
        centroids_new[mask,:] = centroids[mask,:]
        discarded_centroids[mask] += 1
        
        if np_any_axis0(discarded_centroids >= discard_freq):
            centroids_new[discarded_centroids >= discard_freq,:] = np.random.randn(np.sum(discarded_centroids >= discard_freq), D)
            discarded_centroids[discarded_centroids >= discard_freq] = 0

        centroids = centroids_new

        for i in range(centroids.shape[0]):
            centroids[i,:] = centroids[i,:]/np.sqrt(centroids[i,:]@centroids[i,:])

    return centroids, dist, predictions


