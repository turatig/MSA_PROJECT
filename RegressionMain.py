"""
Main program simply calls functions from the Experiment module
"""
import pandas as pd
import numpy as np
from Preprocessing import *
from CrossValidation import GridSearchCV,NestedCVEstimate
from Plots import *
from RidgeRegression import RidgeRegression
from sklearn.decomposition import PCA
from Experiment import *



def logShuffledCVEstimates(estimates,title):
    print("\n","*"*100,"\n")
    print(title)
    print("Best estimate:")
    print(estimates[0])
    print("Variance of the estimates:")
    print(np.var(estimates))

if __name__=="__main__":

    #### Read dataset and split it into features (X) and labels (y)
    data=pd.read_csv('cal-housing.csv')
    #### Performs one-hot ancoding for categorical values in dataset
    data=pd.get_dummies(data)
    #### Dropping na values if they're less than the 5% of the dataset
    if sum(data.isna().sum()/data.shape[0])<0.05:
        data.dropna(inplace=True)

    X=data.drop('median_house_value',axis=1).to_numpy()
    y=data['median_house_value'].to_numpy()

    
    """
        GridSearch CV plus nested CV estimates and plot to study dependence of the risk estimate
        on hyperparamter alpha.
        estimateRegression return the best estimator found with GridSearchCV
    """
    best=estimateRegression(X,y,1,10000,100)["estimator"]

    """
        Plot target labels and try to shuffle the dataset to verify the realiability of the data
    """
    fig,ax=plt.subplots(1)
    ax.plot(y)
    ax.set_ylabel("Target labels")


    """
        Shuffle dataset to find the reliability of the dataset collected
    """
    fig,ax=plt.subplots(1)
    ax.set_title("Shuffled data")
    estimates=shuffledEstimate(best,X,y,ax)
    estimates.sort(reverse=True)
    logShuffledCVEstimates(estimates,"Shuffle dataset")

    """
        Standardize data before computing estimates
    """
    fig,ax=plt.subplots(1)
    ax.set_title("Shuffled dataset and standardized features")
    estimates=shuffledEstimate(RidgeRegression(alpha=best.getAlpha(),
                                    fit_intercept=best.getFitIntercept,
                                    transformer=StdScaler()),X,y,ax)
    estimates.sort(reverse=True)
    logShuffledCVEstimates(estimates,"Shuffle dataset and standardize features")
    """
        Display correlation matrix to identify correlated features
    """
    
    print("Correlation matrix:")
    print(np.corrcoef(X))

    pca=PCA().fit(X)
    fig,ax=plt.subplots(1)
    ax.set_ylabel("Singular values")
    ax.plot(pca.singular_values_)

    fig,ax=plt.subplots(1)
    ax.set_title("Dimensionality reduction to 2 components")
    pca=PCA(n_components=2)
    estimates=shuffledEstimate(RidgeRegression(alpha=best.getAlpha(),
                                    fit_intercept=best.getFitIntercept,
                                    transformer=StdScaler()),X,y,ax)
    
    estimates.sort(reverse=True)
    logShuffledCVEstimates(estimates,"PCA 2 components")
    plt.show()