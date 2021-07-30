import numpy as np

def sigmoid_of(z):
    return 1 / (1 + np.exp(-z))
