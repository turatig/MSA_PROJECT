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

    reliableData(best,X,y)

    plt.show()
    exit()

    #### Use PCA to compute the same estimates. Select the number of components that count for 95% of the total variance
    pca=PCA(0.95,whiten=True)
    pca=pca.fit(stdScale(X))
    X=pca.transform(stdScale(X))

    #### Principal components
    estimateRegression(X,stdScale(y),1,10000,100)
    estimateRegression(X,stdScale(y),0.05,1,100)

    plt.show()

    
