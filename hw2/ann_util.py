import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-1*X))
    #return np.tanh(X)
    #return np.maximum(X, 0)
def sigprime(X) :
	return sigmoid(X) * (1 - sigmoid(X))
    #return 1 - sigmoid(X)**2
    #return X > 0
