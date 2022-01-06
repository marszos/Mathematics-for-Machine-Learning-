import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
n_features = 2
mu = 10
sigma = 2.5

np.random.seed(42)
X = np.random.normal(mu, sigma, size=(n_samples, n_features))



def covariance(X,Y):
    
    xbar = np.mean(X)
    ybar = np.mean(Y)
    
    
    cov = 1 / (len(X) - 1) * np.sum((X - xbar)*(Y - ybar))
    
    return cov


def covariance_matrix(matrix):
    n_features = matrix.T.shape[0]
    
    C = np.zeros([n_features, n_features])
    
    for c_row, feature1 in enumerate(matrix.T):
        for c_column, feature2 in enumerate(matrix.T):
            C[c_row, c_column] = covariance(feature1, feature2)
    
    return C
