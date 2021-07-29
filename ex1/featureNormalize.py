import numpy as np

def feature_normalize(x_data):
    mu = np.mean(x_data, 0)
    std = np.std(x_data, 0)

    x_data -= mu
    x_data /= std

    return x_data, mu, std


