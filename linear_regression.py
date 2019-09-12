import numpy as np
import pandas as pd


def mean_square_error(w, X, y):


    y_pred = X.dot(w)
    err = np.mean(np.square(y_pred - y))
    return err


def linear_regression_noreg(X, y):

    w = []

    for i in range(len(X[0])):
        w.append(np.linalg.inv(X.dot(X.T)) * X.dot(y))

    return w


def linear_regression_invertible(X, y):

    w = None
    return w

def regularized_linear_regression(X, y, lambd):
  
    w = None
    return w


def tune_lambda(Xtrain, ytrain, Xval, yval):
   
    bestlambda = None
    return bestlambda
    

def mapping_data(X, power):

    return X


