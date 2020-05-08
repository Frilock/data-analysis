import pandas as pd
import numpy as np
from sklearn import impute, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import utils as utils


def filling(file_name):
    data_frame = pd.read_csv(file_name, delimiter=',')
    data_frame = pd.DataFrame(data_frame).replace(' ?', np.nan)
    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data_frame = pd.DataFrame(imputer.fit_transform(data_frame))
    if len(data_frame.columns) == 15:
        del data_frame[14]
    return data_frame


def sample_encoder(x_train):
    encoder = preprocessing.LabelEncoder()
    for i in x_train.columns:
        if x_train[i].dtype == object:
            x_train[i] = encoder.fit_transform(x_train[i])
        else:
            pass
    return x_train


def svm_classifier(x_train, y_train, x_test):
    clf = SVC()

    train_x, train_y, validate_x, validate_y = utils.sample_delimiter(0.70, sample_encoder(x_train), y_train)

    clf.fit(train_x, train_y)
    print("accuracy:", metrics.accuracy_score(validate_y, clf.predict(validate_x)))
    return clf.predict(sample_encoder(x_test))


def forest_classifier(x_train, y_train, x_test):
    clf = RandomForestClassifier(n_estimators=100)

    train_x, train_y, validate_x, validate_y = utils.sample_delimiter(0.70, sample_encoder(x_train), y_train)

    clf.fit(train_x, train_y)
    print("accuracy:", metrics.accuracy_score(validate_y, clf.predict(validate_x)))
    return clf.predict(sample_encoder(x_test))


def main():
    train_data_frame = filling('../../csv/classifier/binary/train2.csv')
    test_data_frame = filling('../../csv/classifier/binary/test2.csv')
    y_train = filling('../../csv/classifier/binary/train_y.csv')

    y_test = forest_classifier(train_data_frame, y_train.values.ravel(), test_data_frame)
    svm_classifier(train_data_frame, y_train.values.ravel(), test_data_frame)

    np.savetxt('../../csv/classifier/binary/lab3.csv', y_test, fmt='%d', delimiter=',')


main()
