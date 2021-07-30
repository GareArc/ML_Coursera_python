import numpy as np
import sigmoid

def cost_function(theta_flattened, x_data, y_data):
    theta = theta_flattened.reshape((-1, 1))
    m = y_data.shape[0]
    h = sigmoid.sigmoid_of(np.dot(x_data, theta))
    return np.sum(np.multiply(-y_data, np.log(h)) - np.multiply(1 - y_data, np.log(1 - h))) / m


# adjusted result for scipy.op.minimize
def gradient(theta_flattened:np.ndarray, x_data, y_data):
    theta = theta_flattened.reshape((-1, 1))
    m = y_data.shape[0]

    h = sigmoid.sigmoid_of(np.dot(x_data, theta))
    grad = np.dot(x_data.T, h - y_data) / m
    return grad
