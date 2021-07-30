import numpy as np
import costFunction

def gradient_descent(x_data, y_data, theta, alpha, iterations):
    m = y_data.shape[0]
    cost_history = np.zeros((iterations, 1))
    for i in range(iterations):
        theta -= alpha * costFunction.gradient(theta, x_data, y_data)
        cost_history[i, :] = costFunction.cost_function(theta, x_data, y_data)
    return theta, cost_history
