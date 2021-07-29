import numpy as np

def normal_equation(x_data:np.ndarray, y_data:np.ndarray):
    return np.dot(np.dot(np.linalg.inv(np.dot(x_data.T ,x_data)), x_data.T), y_data)
