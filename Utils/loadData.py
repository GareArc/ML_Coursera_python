import numpy as np
from scipy.io import loadmat  # read matfile

# Load txt x, y data from given txt files. Return numpy matrix or vector object.
# MUST provide absolute path.
def load_data_txt(pathname: str):
    file = open(pathname)
    lines = file.readlines()
    m = len(lines)
    num_features = len(lines[0].split(",")) - 1

    x_data = np.zeros([m, num_features], dtype=float)
    y_data = np.zeros([m, 1], dtype=float)
    for i in range(m):
        line_segs = lines[i].split(",")
        x_data[i, :] = line_segs[: num_features]
        y_data[i, :] = line_segs[-1]
    file.close()
    return x_data, y_data

# Load .mat file. Return numpy matrix or vector object.
# pre: the mat file ONLY contains two variables.
def load_data_mat(pathname: str, x_name: str="X", y_name: str="y"):
    data = loadmat(pathname)
    x_data = data.get(x_name)
    y_data = data.get(y_name)

    return x_data, y_data

