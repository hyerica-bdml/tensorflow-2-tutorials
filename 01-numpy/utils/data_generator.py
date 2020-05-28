import numpy as np
import matplotlib.pyplot as plt


def generate_simple_house_price_dataset():
    area = np.random.normal(loc=25, scale=5, size=100)
    price = np.random.normal(loc=area*300, scale=500, size=100)

    return area, price
