import numpy as np
import itertools as it
from RidgeRegression import RidgeRegression
from Metrics import *

#### Yield k test folds and the corresponding train part one by one
def kFoldIterator(X,y,k):
    #### (X.shape[0]+k-1)//k : add k-1 to handle train set which size m is not divisible for k
    m=X.shape[0]+k-1

    for i in range(k):
        #### Train part of the k-fold
        trainX=np.concatenate( [X[ (i-1)*(m//k) : i*(m//k) ], 
                                X[ (i+1)*(m//k) : ]] )

        trainY=np.concatenate( [y[ (i-1)*(m//k) : i*(m//k) ], 
                                y[ (i+1)*(m//k) : ]] )
        
        #### Test part of the k-fold
        testX=X[ i*(m//k) : (i+1)*(m//k) ]
        testY=y[ i*(m//k) : (i+1)*(m//k) ]

        yield {"train": {"data":trainX,"target":trainY}, 
                "test": {"data":testX,"target":testY}
            }

"""
    Perform a grid search cross validation.
    Return a listed of scores so
"""
def GridSearchCV(estimator,hparams,X,y,k=5,metric=mse):
    
    scoresList=[]
    m=X.shape[0]

    #### Testing on any combination of the params
    for combination in list(it.product(*hparams.values())):

        #### Setting the hyperparam of the algorithm
        h={d[0] : d[1] for d in zip(hparams.keys(),combination)}
        
        estimator=RidgeRegression()
        estimator.set_params(**h)

        testErr=[]
        for fold in kFoldIterator(X,y,k):
            estimator=estimator.fit(fold["train"]["data"],fold["train"]["target"])

            #### Compute the performance on test fold with the given metric
            testErr.append( metric (estimator.predict(fold["test"]["data"]),
                            fold["test"]["target"]) )

            """pred=estimator.predict(fold["test"]["data"])
            testErr.append( k/X.shape[0] * sum((pred-fold["test"]["target"])**2) )"""

        #### Computing mean test error and variance of predictors
        mean=1/k*sum(testErr)
        var=(1/k)*(1/(len(testErr)-1))*sum([(i-mean)**2 for i in testErr])

        estimator=estimator.fit(X,y)
        scoresList.append({"estimator": estimator,"meanScore":mean,"variance":var})

    scoresList.sort(key=lambda e:e["meanScore"])
    return scoresList
            
#### Perform nested cross validation estimate
def NestedCVEstimate(estimator,hparams,X,y,k,metric="mse"):
    estimatedRisk=[]
    m=X.shape[0]

    for fold in kFoldIterator(X,y,k):
        #### Grid search to find best hyperParams for test fold
        scoresList=GridSearchCV(estimator,hparams,fold["test"]["data"],fold["test"]["target"],k)

        """Train the algorithm on the train part of the fold with the best hyperParams 
            found with internal CV
        """
        estimator.set_params(**scoresList[0]["estimator"].get_params())
        estimator=estimator.fit(fold["train"]["data"],fold["train"]["target"])

        estimatedRisk.append( metric (estimator.predict(fold["test"]["data"]),
                            fold["test"]["target"]))

    return (1/k)*sum(estimatedRisk)

    

