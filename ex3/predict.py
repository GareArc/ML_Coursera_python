import numpy as np
import sigmoid

def predict_single_lr(theta_all ,x_data):
    x_data = x_data.reshape((1, -1))

    # add first column
    x_data = np.column_stack((np.ones((1, 1)), x_data))
    result = sigmoid.of(x_data @ theta_all.T)

    predict = np.argmax(result.flatten())
    if predict == 10:
        predict = 0
    return predict

def predict_lr_accuracy(theta_all ,x_data, y_data):
    correct = 0
    m = y_data.shape[0]
    for i in range(m):
        pred = predict_single_lr(theta_all, x_data[i, :])
        if pred == y_data[i, :]:
            correct += 1
    return correct / m

def predict_nn(theta1, theta2, x_data):
    x_row = x_data.shape[0]
    a1 = x_data
    a1 = np.column_stack((np.ones((x_row, 1)), a1))
    z2 = a1 @ theta1.T
    a2 = sigmoid.of(z2)
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2 @ theta2.T
    a3 = sigmoid.of(z3)
    return np.argmax(a3, axis=1) + 1 # add one to match y_data because python starts index from 0.
