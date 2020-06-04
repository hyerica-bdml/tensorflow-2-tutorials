import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn


def generate_simple_house_price_dataset():
    area = np.random.normal(loc=25, scale=5, size=100)
    price = np.random.normal(loc=area*300, scale=500, size=100)

    return area, price


def generate_simple_dataset():
    x = np.zeros((100, 5))
    cov1 = np.eye(5)
    cov2 = np.eye(5)
    
    for i in range(5):
        cov1[i, i] = np.random.rand() * 10
        cov2[i, i] = np.random.rand() * 5
    
    x[:50] = mvn.rvs(mean=[10, 10, 10, 10, 10], cov=cov1, size=(50,))
    x[50:] = mvn.rvs(mean=[0, 0, 0, 0, 0], cov=cov2, size=(50,))
    
    return x
