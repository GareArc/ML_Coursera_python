import numpy as np
import matplotlib.pyplot as plt
import displayData
import lrCostFunction
import oneVsAll
import predict
from Utils import loadData

# ===================== 1. Load Data =====================
x_data, y_data = loadData.load_data_mat("F:\Github repo\Python\ML_Coursera_python\ex3\ex3data1.mat")

# hint : must transpose the data to get the oriented data
x_data = np.array([im.reshape((20, 20)).T for im in x_data])
x_data = np.array([im.reshape((400, )) for im in x_data])
# print(x_data[2, :])
# set some params
input_layer_size = 400
num_labels = 10
# ================== visualize the data ====================================================
rand = np.random.randint(0, 5000, (100, ))  # [0, 5000)
# displayData.display_data(x_data[rand, :])   # get 100 images randomly

# ======================= Test case for lrCostFunction =============================
theta_t = np.array([-2, -1, 1, 2])
t = np.linspace(1, 15, 15) / 10
t = t.reshape((3, 5))
x_t = np.column_stack((np.ones((5, 1)), t.T))
y_t = np.array([1, 0, 1, 0, 1])
l_t = 3
cost = lrCostFunction.get_cost_reg(theta_t, x_t, y_t, l_t)
grad = lrCostFunction.get_gradient_reg(theta_t, x_t, y_t, l_t)
print("cost is {}".format(cost))
print("expected cost is 2.534819")
print("grad is {}".format(grad))
print("expected grad is 0.146561 -0.548558 0.724722 1.398003")

# ============================ one vs all:predict ===========================================
l = 0.1
theta = oneVsAll.one_vs_all(x_data, y_data, l, num_labels)
result = predict.predict_single_lr(theta, x_data[1200, :])
np.set_printoptions(precision=2, suppress=True)  # don't use  scientific notation
print("this number is {}".format(result))  # 10 here is 0
plt.imshow(x_data[1200, :].reshape((20, 20)), cmap='gray', vmin=-1, vmax=1)

accuracy = predict.predict_lr_accuracy(theta, x_data, y_data)
print("test 5000 images, accuracy is {:%}".format(accuracy))

plt.show()
