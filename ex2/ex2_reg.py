import numpy as np
import plotData_reg
import mapFeature
import costFunctionReg
import gradientDescentReg
import matplotlib.pyplot as plt
import plotDecisionBoundary
import scipy.optimize as op

import sys
sys.path.append("..")
from Utils import loadData

# ==================== 1. Load Data ======================
x_data, y_data = loadData.load_data_txt("F:\Github repo\Python\ML_Coursera_python\ex2\ex2data2.txt")


# ======================== 2. Plot Data =======================
# plotData_reg.plot_data(x_data, y_data)

# ================== 3. regularized logistic regression ==============
# get polynomial of x_data
x_data_new = mapFeature.map_feature(x_data[:, 0], x_data[:, 1])
# add first column
x_data_new = np.column_stack((np.ones((x_data_new.shape[0], 1)), x_data_new))
# ================== 4. costFunctionReg ==========================
lambd = 10
theta = np.ones((x_data_new.shape[1], 1))

grad = costFunctionReg.get_gradient(theta, x_data_new, y_data, lambd)
print("grad is {}".format(grad[0:5]))
print("Expected 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n")

# ================== 5. gradient descent ==========================
alpha = 0.1
iterations = 10000
lambd = 0.01
theta = np.ones((x_data_new.shape[1], 1))

# # gradient descent=================
# theta, cost_history_2 = gradientDescentReg.gradient_descent(x_data_new, y_data, theta, alpha, lambd, iterations)
#
# # draw cost-iteration diagram
# plt.figure()
# plt.plot(np.linspace(0, iterations, iterations), cost_history_2)

# scipy
theta = op.minimize(costFunctionReg.get_cost, theta,
                    args=(x_data_new, y_data, lambd),
                    method="TNC",
                    jac=costFunctionReg.get_gradient_for_scipy).x


# draw boundary diagram
plotDecisionBoundary.plot_decision_boundary(theta, x_data_new, y_data)

# Show all plots
plt.show()
