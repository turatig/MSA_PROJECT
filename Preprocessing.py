"""
This module provides function to preprocess data
"""
import numpy as np
import numpy.random as rn

#### Center data points in mat A by subtracting the mean
def center(X): return X-np.average(X,axis=0)

#### Scale data points to have them with variance=1
def scale(X): return X/np.std(X,axis=0)

#### Return (X-Xm)/Xstd
def stdScale(X): return scale(center(X))

#### Util to shuffleDataset
def shuffleDataset(X,y):
    y=np.array([[e] for e in y])
    a=np.concatenate([X,y],axis=1)
    rn.shuffle(a)
    return np.array([row[:-1] for row in a]),np.array([row[-1] for row in a])

#### Class that implements a simple standard scaler
class StdScaler():

    def __init__(self):
        self.mean=0.
        self.std_dev=0.

    def fit(self,X):
        self.mean=np.average(X,axis=0)
        self.std_dev=np.std(X,axis=0)
        return self

    def transform(self,X):
        return (X-self.mean)/self.std_dev

    