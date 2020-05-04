import pandas as pd
import numpy as np
from sklearn import svm, impute, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing

# пропускается первая строка при считывании файла


def filling(file_name):
    data_frame = pd.read_csv(file_name, delimiter=',')
    data_frame = pd.DataFrame(data_frame)
    data_frame = data_frame.replace(' ?', np.nan)

    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    data_frame = imputer.fit_transform(data_frame)
    data_frame = pd.DataFrame(data_frame)

    if len(data_frame.columns) == 15:
        del data_frame[14]

    return data_frame


def forest_classifier(x_train, y_train, x_test):
    clf = RandomForestClassifier(n_estimators=100)

    encoder = preprocessing.LabelEncoder()
    for i in x_train.columns:
        if x_train[i].dtype == object:
            x_train[i] = encoder.fit_transform(x_train[i])
        else:
            pass
    for j in x_test.columns:
        if x_test[j].dtype == object:
            x_test[j] = encoder.fit_transform(x_test[j])
        else:
            pass

    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def main():
    train_data_frame = filling('../csv/classifier/binary/train2.csv')
    # print(train_data_frame)
    test_data_frame = filling('../csv/classifier/binary/test2.csv')
    # print(test_data_frame)
    y_train = filling('../csv/classifier/binary/train_y.csv')
    # print(y_train)

    y_test = forest_classifier(train_data_frame, y_train.values.ravel(), test_data_frame)
    # print(y_test)
    np.savetxt('../csv/classifier/binary/lab2_1.csv', y_test, fmt='%d', delimiter=',')


main()
