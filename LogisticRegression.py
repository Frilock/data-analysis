import numpy as np
import math
import utils as utils


class LogisticRegression(object):
    def __init__(self, step_size, count_iterations):
        self.step_size = step_size
        self.count_iterations = count_iterations
        self.train_x = np.genfromtxt('csv/regression/t1_logreg_x_train.csv', delimiter=',')
        self.train_y = np.genfromtxt('csv/regression/t1_logreg_y_train.csv', delimiter=',').reshape((-1, 1))
        self.test_x = np.genfromtxt('csv/regression/t1_logreg_x_test.csv', delimiter=',')

    def gradient_descent(self):
        x = utils.prepare_x(self.test_x)

        train_percent = int(len(x) * 0.7)
        x_train = x[:train_percent]
        x_validate = x[:len(x) - train_percent]
        y_train = self.train_y[:train_percent].reshape((-1, 1))
        y_validate = self.train_y[:len(x) - train_percent].reshape((-1, 1))

        curr_theta = np.zeros((len(x[0]), 1))
        prev_theta = np.random.rand(len(x[0]), 1)

        for step in range(self.count_iterations):
            a = np.zeros((len(x_train), 1))
            for i in range(len(x_train)):
                a[i] = self.derivative_func(x_train[i], prev_theta)

            curr_theta = np.subtract(prev_theta,
                                     self.step_size * np.dot(np.transpose(x_train), np.subtract(a, y_train)))
            prev_theta = curr_theta

        self.accuracy_calculation(x_validate, y_validate, curr_theta)

        return curr_theta

    def accuracy_calculation(self, x_validate, y_validate, current_theta):
        accuracy = 0
        current_y = self.predict(x_validate, current_theta)
        for v1, v2 in zip(y_validate, current_y):
            if v1 == v2:
                accuracy += 1
        print(accuracy / len(y_validate))

    def predict(self, x, theta):
        current_y = np.zeros((len(x), 1), dtype=int)
        for i in range(len(current_y)):
            if self.derivative_func(x[i], theta) > 0.5:
                current_y[i] = 1
            else:
                current_y[i] = 0
        return current_y

    @staticmethod
    def derivative_func(x, theta):
        return 1 / (1 + math.exp(-1 * x.dot(theta)))
