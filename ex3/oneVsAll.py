import numpy as np
import fmincg

def one_vs_all(x_data, y_data, lam, num_labels):
    # initialize some params
    m, n = x_data.shape
    x_data = np.column_stack((np.ones((m, 1)), x_data))
    # initialize initial_theta
    initial_theta = np.zeros((num_labels, n+1))
    theta = fmincg.fmincg(initial_theta, x_data, y_data, lam, num_labels)
    return theta
