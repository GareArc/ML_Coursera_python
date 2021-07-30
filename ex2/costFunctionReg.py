import numpy as np
import sigmoid

def get_cost(theta, x_data, y_data, lam):
    m = y_data.shape[0]
    theta = theta.reshape(-1, 1)

    h = sigmoid.sigmoid_of(np.dot(x_data, theta))
    part1 = np.sum(np.multiply(-y_data, np.log(h)) - np.multiply(1 - y_data, np.log(1 - h))) / m
    part2 = lam * np.dot(theta[1:, :].T, theta[1:, :]) / 2 / m
    return part1 + part2


def get_gradient(theta, x_data, y_data, lam):
    m = y_data.shape[0]

    theta_copy = theta.reshape(-1, 1).copy()
    h = sigmoid.sigmoid_of(np.dot(x_data, theta_copy)) # theta_0 is not 0 here.
    theta_copy[0, 0] = 0
    return (np.dot(x_data.T, h - y_data) / m) + (lam * theta_copy / m) # since theta_0 is 0, the second part becomes 0 for the first row.

def get_gradient_for_scipy(theta, x_data, y_data, lam):
    return get_gradient(theta, x_data, y_data, lam).flatten()

