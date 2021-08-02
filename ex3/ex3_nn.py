import numpy as np
import matplotlib.pyplot as plt
from Utils import loadData
import predict
import displayData

# =========== Part 1: Loading and Visualizing Data =============
x_data, y_data = loadData.load_data_mat("F:\Github repo\Python\ML_Coursera_python\ex3\ex3data1.mat")
input_layer_size = 400
hidden_layer_size = 25   # 25 hidden units
num_labels = 10

# we must transpose x_data to show the image correctly
x = np.array([im.reshape((20, 20)).T for im in x_data])
x = np.array([im.reshape((400, )) for im in x])
rand = np.random.randint(0, 5000, (100, ))  # [0, 5000)

displayData.display_data(x[rand, :])   # get 100 images randomly

# ================ Part 2: Loading Parameters ================
theta1, theta2 = loadData.load_data_mat("F:\Github repo\Python\ML_Coursera_python\ex3\ex3weights.mat",
                                        "Theta1",
                                        "Theta2")

# ================= predict ==================================
result = predict.predict_nn(theta1, theta2, x_data)
# compute accuracy
right_num = 0
for i in range(result.shape[0]):
    if result[i] == y_data[i, :]:
        right_num += 1
accuracy = right_num / result.shape[0]
print("test 5000 images accuracy is {:%}".format(accuracy))
