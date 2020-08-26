from RidgeRegression import RidgeRegression
from CrossValidation import GridSearchCV as gsc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from numpy.testing import *
import unittest
import pandas as pd
import numpy as np
import random as rnd


def getModels(X,y,alpha,fit_intercept=True):
    return RidgeRegression(alpha,fit_intercept=fit_intercept).fit(X,y),\
            Ridge(alpha,fit_intercept=fit_intercept).fit(X,y)
"""
Implementation tests. The model is fitted in every test case to test for random values of alpha.
Assert methods from numpy are used to avoid that numerical approximation makes tests fail.
"""
class RidgeImplementationTest(unittest.TestCase):

    def setUp(self):
        
        #### Read dataset and split it into features (X) and labels (y)
        self.data=pd.read_csv('cal-housing.csv')
        #### Performs one-hot ancoding for categorical values in dataset
        self.data=pd.get_dummies(self.data)
        #### Dropping na values if they're less than the 5% of the dataset
        if sum(self.data.isna().sum()/self.data.shape[0])<0.05:
            self.data.dropna(inplace=True)

        self.X=self.data.drop('median_house_value',axis=1).to_numpy()
        self.y=self.data['median_house_value'].to_numpy()
    
    #### Testing the coefficients of the predictor
    def test_coefficients(self):

        print("*"*20," COEFFICIENTS TEST ","*"*20)
        alpha=rnd.random()*100

        #### Without computing the intercept of the model
        implEst,skEst=getModels(self.X,self.y,alpha,False)
        assert_array_almost_equal(implEst.w,skEst.coef_)

        #### Fitting the intercept
        implEst,skEst=getModels(self.X,self.y,alpha)
        assert_array_almost_equal(implEst.w,skEst.coef_)

        print("*"*20," PASSED ","*"*20)

    #### Testing the intercept of the model
    def test_intercept(self):

        print("*"*20," INTERCEPT TEST ","*"*20)
        alpha=rnd.random()*100

        #### Fitting the intercept
        implEst,skEst=getModels(self.X,self.y,alpha)
        assert_almost_equal(implEst.intercept,skEst.intercept_,decimal=5)

        print("*"*20," PASSED ","*"*20)

    #### Testing predictions
    def test_prediction(self):

        print("*"*20," PREDICTION TEST ","*"*20)
        alpha=rnd.random()*100
        randomPoint=np.array([[rnd.random()*100 for i in range(self.X.shape[1])]])

        #### Without computing the intercept of the model
        implEst,skEst=getModels(self.X,self.y,alpha,False)
        assert_array_almost_equal(implEst.predict(randomPoint),skEst.predict(randomPoint),decimal=5)

        #### Fitting the intercept
        implEst,skEst=getModels(self.X,self.y,alpha)
        assert_array_almost_equal(implEst.predict(randomPoint),skEst.predict(randomPoint),decimal=5)

        print("*"*20," PASSED ","*"*20)

    #### Testing crossValidation estimate
    def test_crossValidation(self):
        start,stop,n_values=10,10000,100
        implScoresList=gsc(RidgeRegression(),{"alpha": np.linspace(start,stop,n_values)},
                                                self.X,self.y,metric="mse")

        skScoresList=GridSearchCV(Ridge(),{"alpha": np.linspace(start,stop,n_values)},scoring="neg_mean_squared_error").\
                                    fit(self.X,self.y)
        
        print(skScoresList.best_estimator_.coef_)
        print(skScoresList.best_params_)
        print(skScoresList.best_score_)
        print(implScoresList[0])


if __name__=="__main__":
   unittest.main()



