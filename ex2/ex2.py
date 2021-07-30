import numpy as np
import costFunction
import plotData
import gradientDescent
import plotDecisionBoundary
import scipy.optimize as op
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from Utils import loadData

plt.clf() # 清图。
plt.cla() # 清坐标轴。
plt.close("all") # 关窗口

# ============================== 1. Load Data ===============================
x_data, y_data = loadData.load_data_txt("F:\Github repo\Python\ML_Coursera_python\ex2\ex2data1.txt")

# ============================== 2. Plot Data ===============================
# plotData.plot(x_data, y_data)

# ============================== 3. compute cost and gradient ===============================
num_of_features = x_data.shape[1]
# Add 1's column
x_data = np.column_stack((np.ones((x_data.shape[0], 1)), x_data))
init_theta = np.zeros((x_data.shape[1], 1))

grad = costFunction.gradient(init_theta, x_data, y_data)
print("initial_theta grad is:\n {}".format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')


# ============= 4. Optimizing using fminunc(matlab)/scipy(python)  =============
result = op.minimize(costFunction.cost_function,
                     init_theta,
                     args=(x_data, y_data),
                     method="TNC",
                     jac=costFunction.gradient).x

# ax1 = plt.subplot(3, 1, 1)
# ax2 = plt.subplot(3, 1, 3)
plotDecisionBoundary.plot_decision_boundary(result, x_data, y_data)

# ========================== 5. Gradient descent ================================
# Gradient descent settings
alpha = 0.0001
iterations = 50
theta = np.zeros((x_data.shape[1], 1))

theta, cost_history = gradientDescent.gradient_descent(x_data, y_data, theta, alpha, iterations)

# print(cost_history)
plt.figure()
plt.plot(np.linspace(0, iterations, iterations), cost_history)
plt.show()
