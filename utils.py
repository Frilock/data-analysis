import numpy as np


def prepare_x(x):
    x_std = x.std(axis=0)
    x_mean = x.mean(axis=0)
    x = (x - x_mean) / x_std
    return np.column_stack((np.ones(len(x)), x))


def sample_delimiter(percent, x_train, y_train):
    train_percent = int(len(x_train) * percent)
    validate_percent = int(len(x_train) - train_percent)
    train_x = x_train[:train_percent]
    train_y = y_train[:train_percent].reshape((-1, 1))
    validate_x = x_train[-validate_percent:]
    validate_y = y_train[-validate_percent:].reshape((-1, 1))

    return train_x, train_y, validate_x, validate_y
