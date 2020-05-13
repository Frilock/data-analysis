import numpy as np
from python import utils as utils
from python.regression.linearRegression import LinearRegression
from python.regression.logisticRegression import LogisticRegression

linearRegression = LinearRegression(0.0001, 100000, 0.70)
logisticRegression = LogisticRegression(0.001, 10000, 0.70)

lin_reg_theta = linearRegression.gradient_descent()
log_reg_theta = logisticRegression.gradient_descent()

lin_reg_x_test = utils.prepare_x(linearRegression.test_x)
lin_reg_y_test = np.dot(lin_reg_x_test, lin_reg_theta)
log_reg_x_test = utils.prepare_x(logisticRegression.test_x)
log_reg_y_test = logisticRegression.predict(log_reg_x_test, log_reg_theta)

np.savetxt('../resources/csv/regression/lab1_1.csv', lin_reg_y_test, delimiter=',')
np.savetxt('../resources/csv/regression/lab1_2.csv', log_reg_y_test, fmt='%d', delimiter=',')
