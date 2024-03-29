"""
MAIN EXPERIMENT PROGRAM
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
    print(data.info())
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
    scoresList=estimateRegression(X,y,1,10000,100)
    best=scoresList[0]["estimator"]
    
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
    estimates=shuffledCVEstimate(best,X,y)
    estimates.sort(reverse=True)
    logShuffledCVEstimates(estimates,"Shuffle dataset")

    """
        Standardize data before computing estimates
    """
    estimates_std=shuffledCVEstimate(Pipe(
            [StdScaler()],RidgeRegression(alpha=best.getAlpha(),fit_intercept=best.getFitIntercept)
        ),
        X,y)
    estimates_std.sort(reverse=True)
    logShuffledCVEstimates(estimates_std,"Shuffle dataset and standardize features")
    ax.plot(estimates)
    ax.plot(estimates_std)
    ax.legend(["Non standardized","Standardized"])
    """
        Display correlation matrix to identify correlated features
    """
    
    plt.matshow(np.corrcoef(X.transpose()))
    plt.colorbar()
    pca=PCA().fit(X)
    fig,ax=plt.subplots(1)
    ax.set_ylabel("Singular values")
    ax.plot(pca.singular_values_)

    
    pca=PCA(n_components=4).fit(X)
    print("\n","*"*100,"\n")
    print("The ratio of variance explained by 4 components is:")
    print(sum(pca.explained_variance_ratio_))

    fig,ax=plt.subplots(1)

    """
    Compute GridSearchCV again with two preprocessing pipelines and compare results
    """
    scoresList1=GridSearchCV(Pipe([PCA(4)],RidgeRegression()),{"alpha": np.linspace(1,10000,100)},
                                                X,y)
    
    scoresList2=GridSearchCV(Pipe([StdScaler(),PCA(4)],RidgeRegression()),{"alpha": np.linspace(1,10000,100)},
                                                X,y)
    ax.set_title("PCA comparison")
    #### Selectin only estimators with fit_intercept=true to do comparisons
    plotTestErr(ax,[el for el in scoresList if el["estimator"].getFitIntercept()==True])
    plotTestErr(ax,scoresList1)
    plotTestErr(ax,scoresList2)
    ax.legend(["Simple","PCA","Standardized PCA"])
    plt.show()