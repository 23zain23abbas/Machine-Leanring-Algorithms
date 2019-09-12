import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):

    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    Xtranspose = X.T

    if loss == "perceptron":
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0

        y = np.where(y > 0.5, 1.0, -1.0)

        for i in range(max_iterations):

            grad_w = Xtranspose.dot(y * np.where(y * np.sign(X.dot(w) + b) <= 0, 1, 0)) / N
            grad_b = np.sum(y * np.where(y * np.sign(X.dot(w) + b) <= 0, 1, 0)) / N
            
            w = w + step_size * grad_w
            b = b + step_size * grad_b


        

    elif loss == "logistic":
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        
        for i in range(max_iterations):
            grad_w = Xtranspose.dot(sigmoid(X.dot(w) + b) - y) / N
            grad_b = np.sum(sigmoid(X.dot(w) + b) - y) / N

            w = w - step_size * grad_w
            b = b - step_size * grad_b

        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    

    value = 1 / (1 + np.exp(-z))
    
    return value

def binary_predict(X, w, b, loss="perceptron"):

    N, D = X.shape
    
    if loss == "perceptron":
        #          Compute preds                   #
        preds = np.zeros(N)
        for i in range(N):
            p = np.dot(X[i],w) + b
            if p > 0:
                preds[i] = 1.0
            else:
                preds[i] = 0.0
        

    elif loss == "logistic":
        ############################################
        #          Compute preds                   #
        preds = np.zeros(N)
        for i in range(N):
            p = np.dot(X[i], w) + b

            if sigmoid(p) > 0.5:
                preds[i] = 1.0
            else:
                preds[i] = 0.0
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):


    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        

    elif gd_type == "gd":
        ############################################
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):

    N, D = X.shape
    ############################################
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################

    assert preds.shape == (N,)
    return preds




