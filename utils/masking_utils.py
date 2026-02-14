import numpy as np
import pandas as pd


def make_proximity_mask(arr, tol, thresholding='hard', multiplier=None):
    
    arr = arr.reshape(len(arr),1)
    D = np.abs(arr - arr.T)
    mask = D <= tol
    if thresholding == 'soft':
        if(multiplier is None):
            multiplier = 1
        
        relu_ = np.vectorize(np.maximum)
        relu = lambda x: relu_(x,0)
        
        mask -= multiplier * np.clip(D,0,tol)
        mask = relu(mask)
        
    return mask

def sortedMask2assignments(mask):
    n = mask.shape[0]
    assignments = np.zeros((n,))
    k=0
    bound=-1
    for i in range(n):
        if i<bound:
            continue
        for j in range(i, n):
            if mask[i,j]==1:
                assignments[j]=k
            else:
                k+=1
                break
        bound=j
    return assignments

def mask2assignments(mask):
    n = mask.shape[0]
    assignments = np.zeros((n,))
    k=0
    assigned=[]
    for i in range(n):
        if i in assigned:
            continue
        for j in range(i, n):
            if mask[i,j]==1:
                assignments[j]=k
                assigned.append(j)    
        k+=1
    return assignments