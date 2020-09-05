"""
This module provides functions for the evaluation of a predictor
"""
import numpy as np

#### Functions to evaluate the performance of a predictor
#### yp: predicted labels
#### y: target labels

#### Mean squared error
def mse(yp,y): return (1/y.shape[0])*sum( (yp-y)**2 )

#### Accuracy
def r2(yp,y): return 1-sum( (y-yp)**2 )/sum((y-np.average(y))**2)
