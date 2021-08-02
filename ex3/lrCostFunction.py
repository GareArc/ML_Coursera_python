import numpy as np
import sigmoid


def get_cost_reg(theta, x_data, y_data, lambd):
    theta = theta.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)
    m = y_data.shape[0]
    h = sigmoid.of(np.dot(x_data, theta))

    part1 = np.sum(np.multiply(-y_data, np.log(h)) - np.multiply(1 - y_data, np.log(1 - h))) / m
    part2 = lambd * np.dot(theta[1:, :].T, theta[1:, :]) / m / 2

    return part1 + part2

def get_gradient_reg(theta, x_data, y_data, lambd):
    theta = theta.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)
    m = y_data.shape[0]
    h = sigmoid.of(np.dot(x_data, theta))

    part1 = np.dot(x_data.T, h - y_data) / m
    theta_temp = theta.copy()
    theta_temp[0, 0] = 0
    part2 = lambd * theta_temp / m

    return part1 + part2

def get_gradient_reg_flattened(theta, x_data, y_data, lambd):
    return get_gradient_reg(theta, x_data, y_data, lambd).flatten()

