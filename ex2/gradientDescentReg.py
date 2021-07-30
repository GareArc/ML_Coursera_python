import numpy as np
import costFunctionReg

def gradient_descent(x_data, y_data, theta, alpha, lambd, iterations):
    m = y_data.shape[0]
    cost_history = np.zeros((iterations, 1))
    for i in range(iterations):
        theta -= alpha * costFunctionReg.get_gradient(theta, x_data, y_data, lambd)
        cost_history[i, :] = costFunctionReg.get_cost(theta, x_data, y_data, lambd)
    return theta, cost_history
