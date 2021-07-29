import numpy as np
import gradientDescent
import featureNormalize
import normalEqn

import sys
sys.path.append("..")
from Utils import loadData

# ================ Part 1: Feature Normalization ================
x_data, y_data = loadData.load_data_txt("F:\Github repo\Python\ML_Coursera_python\ex1\ex1data2.txt")

# normalize data
x_data, mu, std = featureNormalize.feature_normalize(x_data)

# Add x_0 column. NOTICE that this column is added AFTER data normalization process.
x_data = np.column_stack((np.ones((x_data.shape[0], 1)), x_data))

# =======================Part2. Gradient Descent ===========================
# gradient descent settings
alpha = 0.01
iterations = 5000
theta = np.zeros((x_data.shape[1], 1))
theta, cost_history = gradientDescent.gradient_descent(x_data, y_data, theta, alpha, iterations)
# print(theta)

# =======================Part3. Predict ================
input_data = np.array([1, 1650, 3], dtype=float)
input_data = input_data[np.newaxis, :]
# IMPORTANT: do feature normalization to data as well.
input_data[0, 1:] -= mu
input_data[0, 1:] /= std

print("predict price is {}".format(np.dot(input_data, theta)))

# =======================Part4. Normal Equation ================
x_data, y_data = loadData.load_data_txt("F:\Github repo\Python\ML_Coursera_python\ex1\ex1data2.txt")
x_data = np.column_stack((np.ones((x_data.shape[0], 1)), x_data))
theta = normalEqn.normal_equation(x_data, y_data)

input_data = np.array([1, 1650, 3], dtype=float)
input_data = input_data[np.newaxis, :]

print("predict price using normal equation is {}".format(np.dot(input_data, theta)))
