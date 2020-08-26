import numpy as np

#### Function to evaluate the performance of a predictor
#### yp: predicted labels
#### y: target labels

METRICS={
        "mse": lambda yp,y: mse(yp,y),
        "r2": lambda yp,y: r2(yp,y)
}

#### Mean squared error
def mse(yp,y): return (1/y.shape[0])*sum( (yp-y)**2 )

#### Accuracy
def r2(yp,y): return 1-sum( (y-yp)**2 )/sum((y-np.average(y))**2)
