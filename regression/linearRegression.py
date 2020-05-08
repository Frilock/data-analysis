import numpy as np
import utils as utils


class LinearRegression(object):
    def __init__(self, step_size, count_iterations, percent):
        self.step_size = step_size
        self.count_iterations = count_iterations
        self.percent = percent
        self.train_x = np.genfromtxt('csv/t1_linreg_x_train.csv', delimiter=',')
        self.train_y = np.genfromtxt('csv/t1_linreg_y_train.csv', delimiter=',').reshape((-1, 1))
        self.test_x = np.genfromtxt('csv/t1_linreg_x_test.csv', delimiter=',')

    def gradient_descent(self):
        x = utils.prepare_x(self.train_x)

        x_train, y_train, x_validate, y_validate = utils.sample_delimiter(self.percent, x, self.train_y)

        prev_theta = np.random.rand(len(x[0]), 1)
        current_theta = 0

        for step in range(self.count_iterations):
            a = np.dot(x_train, prev_theta)
            current_theta = np.subtract(prev_theta,
                                        self.step_size * np.dot(np.transpose(x_train), np.subtract(a, y_train)))
            prev_theta = current_theta
        current_y = np.dot(x_validate, current_theta)
        print(self.coefficient_of_determination(y_validate, current_y))

        return current_theta

    @staticmethod
    def coefficient_of_determination(real_y, current_y):
        ssr = np.sum((current_y - real_y) ** 2)
        sst = np.sum((real_y - np.mean(real_y)) ** 2)
        return 1 - (ssr / sst)
