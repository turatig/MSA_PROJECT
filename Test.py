"""
This module contains unit tests to compare this implementation of the Ridge Regression algorithm with 
that one of sklearn
"""
from numpy.testing import *
import unittest

import pandas as pd
import numpy as np
import random as rnd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from Preprocessing import StdScaler
from RidgeRegression import RidgeRegression
from CrossValidation import GridSearchCV as gsc
from Metrics import *


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

    #### Testing standard scaler
    def test_standardScaler(self):
        
        print("*"*20," STANDARD SCALER TEST ","*"*20)
        alpha=rnd.random()*100
        randomPoint=np.array([[rnd.random()*100 for i in range(self.X.shape[1])]])

        #### sklearn results
        pipe=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        pipe=pipe.fit(self.X,self.y)
        skRes=pipe.predict(randomPoint)

        implEst=RidgeRegression(alpha=alpha,transformer=StdScaler()).fit(self.X,self.y)
        res=implEst.predict(randomPoint)

        assert_almost_equal(res,skRes,decimal=4)



if __name__=="__main__":
   unittest.main()



