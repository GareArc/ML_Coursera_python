import numpy as np
import computeCost

def gradient_descent(x_data: np.ndarray,
                     y_data: np.ndarray,
                     theta: np.ndarray, alpha, iterations):
    m = np.shape(y_data)[0]
    cost_history = np.zeros((iterations, 1))

    for i in range(iterations):
        theta -= alpha * np.dot(x_data.T, (np.dot(x_data, theta) - y_data)) / m
        cost_history[i, 0] = computeCost.compute_cost(x_data, y_data, theta)
    return theta, cost_history
