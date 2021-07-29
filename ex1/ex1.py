# From current dir
import warmUpExercise
import plotData
import computeCost
import gradientDescent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# From Util dir
import sys
sys.path.append("..")
from Utils import loadData

# ================== Part 1 warmUpExercise ====================
# warmUpExercise.warm_up_exercise()

# ================== Part 2 Plotting data ====================
x_data, y_data = loadData.load_data_txt("F:\Github repo\Python\ML_Coursera_python\ex1\ex1data1.txt")
plotData.plot(x_data, y_data)

# ================== Part 3 Cost and Gradient descent ====================
m = np.shape(y_data)[0]
# Add first column
x_data_new = np.column_stack((np.ones([m, 1]), x_data))
theta = np.zeros((np.shape(x_data_new)[1], 1))
# Gradient descent settings
iterations = 1500
alpha = 0.01
# Compute cost
J = computeCost.compute_cost(x_data_new, y_data, theta)
print("With theta {}, cost is {}.".format(theta, J))
print("Expected cost is 32.07")
# Gradient descent
theta, _ = gradientDescent.gradient_descent(x_data_new, y_data, theta, alpha, iterations)
print("theta is {}".format(theta))
print("after {} iterations,theta is {}".format(iterations, theta))
print("expected theta is [-3.6303,1.1664]")
# Plot the linear fit
plt.plot(x_data_new[:, 1], np.dot(x_data_new, theta))
plt.scatter(x_data_new[:, 1], y_data, marker="*", edgecolors="red")
plt.show()

# ================== Part 4  Visualizing J(theta_0, theta_1) ====================
theta0_interval = np.linspace(-10, 10, 100)
theta1_interval = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_interval.shape[0], theta1_interval.shape[0]))
for i in range(len(theta0_interval)):
    for ii in range(len(theta1_interval)):
        # loop through theta0 and theta1 and form theta vectors one by one.
        t = np.vstack((theta0_interval[i], theta1_interval[ii]))
        # feed each theta vector to compute cost
        J_vals[i, ii] = computeCost.compute_cost(x_data_new, y_data, t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("Visualizing J(theta_0, theta_1):surface")
ax.plot_surface(theta0_interval, theta1_interval, J_vals.T, cmap="rainbow")
plt.show()

plt.figure()
plt.title("Visualizing J(theta_0, theta_1):contour")
plt.contour(theta0_interval, theta1_interval, J_vals.T, np.logspace(-2, 3, 20), cmap='rainbow')
plt.scatter(theta[0], theta[1], marker="*")
plt.show()
