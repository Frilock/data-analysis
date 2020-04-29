import numpy as np


class LinearRegression(object):
    def __init__(self, step_size, count_iterations):
        self.step_size = step_size
        self.count_iterations = count_iterations
        self.train_x = np.genfromtxt('csv/regression/t1_linreg_x_train.csv', delimiter=',')
        self.train_y = np.genfromtxt('csv/regression/t1_linreg_y_train.csv', delimiter=',').reshape((-1, 1))
        self.test_x = np.genfromtxt('csv/regression/t1_linreg_x_test.csv', delimiter=',')

    def gradient_descent(self):
        # training_errors = []
        # validation_errors = []
        # steps = []
        x = self.prepare_x(self.train_x)

        train_percent = int(len(x) * 0.7)
        x_train = x[:train_percent]
        x_validate = x[:len(x) - train_percent]
        y_train = self.train_y[:train_percent].reshape((-1, 1))
        y_validate = self.train_y[:len(x) - train_percent].reshape((-1, 1))

        prev_theta = np.random.rand(len(x[0]), 1)
        curr_theta = np.zeros((len(x[0]), 1))

        for step in range(self.count_iterations):
            a = np.dot(x_train, prev_theta)

            curr_theta = np.subtract(prev_theta,
                                     self.step_size * np.dot(np.transpose(x_train), np.subtract(a, y_train)))

            # training_error = self.error_calculation(y_train, np.dot(x_train, curr_theta))
            # validation_error = self.error_calculation(y_validate, np.dot(x_validate, curr_theta))

            # training_errors.append(training_error)
            # validation_errors.append(validation_error)

            # steps.append(step)
            prev_theta = curr_theta

        current_y = np.dot(x_validate, curr_theta)
        print(self.coefficient_of_determination(y_validate, current_y))

        return curr_theta

    @staticmethod
    def prepare_x(x):
        x_std = x.std(axis=0)
        x_mean = x.mean(axis=0)
        x = (x - x_mean) / x_std
        x = np.concatenate((np.ones(len(x))[:, np.newaxis], x), axis=1)
        return x

    @staticmethod
    def error_calculation(current_y, real_y):
        return (((real_y - current_y) ** 2) / 2).sum() / len(real_y)

    @staticmethod
    def coefficient_of_determination(real_y, current_y):
        ssr = np.sum((current_y - real_y) ** 2)
        sst = np.sum((real_y - np.mean(real_y)) ** 2)
        r2_score = 1 - (ssr / sst)
        return r2_score
