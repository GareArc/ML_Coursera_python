import numpy as np

def compute_cost(x_data, y_data, theta):
    m = np.shape(y_data)[0]
    return np.sum(np.power((np.dot(x_data, theta) - y_data), 2)) / 2 / m
