import numpy as np
import utils as utils


class LinearRegression(object):
    def __init__(self, step_size, count_iterations):
        self.step_size = step_size
        self.count_iterations = count_iterations
        self.train_x = np.genfromtxt('csv/regression/t1_linreg_x_train.csv', delimiter=',')
        self.train_y = np.genfromtxt('csv/regression/t1_linreg_y_train.csv', delimiter=',').reshape((-1, 1))
        self.test_x = np.genfromtxt('csv/regression/t1_linreg_x_test.csv', delimiter=',')

    def gradient_descent(self):
        x = utils.prepare_x(self.train_x)

        train_percent = int(len(x) * 0.75)
        validate_percent = int(len(x) - train_percent)
        x_train = x[:train_percent]
        y_train = self.train_y[:train_percent].reshape((-1, 1))
        x_validate = x[:validate_percent]
        y_validate = self.train_y[:validate_percent].reshape((-1, 1))

        prev_theta = np.random.rand(len(x[0]), 1)
        current_theta = 0

        for step in range(self.count_iterations):
            temp = np.dot(x_train, prev_theta)
            current_theta = np.subtract(prev_theta,
                                        self.step_size * np.dot(np.transpose(x_train), np.subtract(temp, y_train)))
            prev_theta = current_theta
        current_y = np.dot(x_validate, current_theta)
        print(self.coefficient_of_determination(y_validate, current_y))

        return current_theta

    @staticmethod
    def coefficient_of_determination(real_y, current_y):
        ssr = np.sum((current_y - real_y) ** 2)
        sst = np.sum((real_y - np.mean(real_y)) ** 2)
        return 1 - (ssr / sst)
