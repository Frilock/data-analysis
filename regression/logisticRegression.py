import numpy as np
import math
import utils as utils


class LogisticRegression(object):
    def __init__(self, step_size, count_iterations, percent):
        self.step_size = step_size
        self.count_iterations = count_iterations
        self.percent = percent
        self.train_x = np.genfromtxt('../', delimiter=',')
        self.train_y = np.genfromtxt('csv/t1_logreg_y_train.csv', delimiter=',').reshape((-1, 1))
        self.test_x = np.genfromtxt('csv/t1_logreg_x_test.csv', delimiter=',')

    def gradient_descent(self):
        x = utils.prepare_x(self.train_x)

        x_train, y_train, x_validate, y_validate = utils.sample_delimiter(self.percent, x, self.train_y)

        current_theta = 0
        prev_theta = np.random.rand(len(x[0]), 1)

        for step in range(self.count_iterations):
            a = np.zeros((len(x_train), 1))
            for i in range(len(x_train)):
                a[i] = self.derivative_func(x_train[i], prev_theta)
            current_theta = np.subtract(prev_theta,
                                        self.step_size * np.dot(np.transpose(x_train), np.subtract(a, y_train)))
            prev_theta = current_theta
        print(self.accuracy_calculation(x_validate, y_validate, current_theta))

        return current_theta

    def predict(self, x, theta):
        current_y = np.zeros((len(x), 1))
        for i in range(len(current_y)):
            if self.derivative_func(x[i], theta) > 0.5:
                current_y[i] = 1
            else:
                current_y[i] = 0
        return current_y

    @staticmethod
    def derivative_func(x, theta):
        return 1 / (1 + math.exp(-1 * x.dot(theta)))

    def accuracy_calculation(self, x_validate, y_validate, current_theta):
        accuracy = 0
        current_y = self.predict(x_validate, current_theta)
        for count1, count2 in zip(y_validate, current_y):
            if count1 == count2:
                accuracy += 1
        return accuracy / len(y_validate)
