import scipy.optimize as op
import numpy as np
import lrCostFunction


def fmincg(theta, x_data, y_data, lam, num_labels):

    for i in range(num_labels):
        y_tempt = y_data.copy()  # hint: must copy !
        # pre-treat
        if i != 0:
            pos = np.where(y_data == i)
            neg = np.where(y_data != i)
        else:
            pos = np.where(y_data == 10)
            neg = np.where(y_data != 10)
        y_tempt[pos, :] = 1
        y_tempt[neg, :] = 0
        result = op.minimize(lrCostFunction.get_cost_reg, theta[i, :].T, args=(x_data, y_tempt, lam), method="TNC", jac=lrCostFunction.get_gradient_reg_flattened)
        # print("{} : {}".format(i, result.success))
        theta[i, :] = result.x
    return theta
