# Необходимо научиться классифицировать текстовые запросы пользователей.
#
# Текстовые запросы пользователей сгруппированы по файлам.
# Каждый файл с префиксом train содержит примеры запросы из одного класса.
# Цифра перед префиксом - метка класса. Файл test.txt содержит тестовые запросы.
#
# Провести сравнение различных методов векторизации запросов:
# 1. n-граммы букв;
# 2. TF-IDF;
# 3. усреднение векторов слов.
import numpy as np
import pandas as pd
import python.classifier.text.textUtils as utils

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


def main():
    # create_x_and_y_train()
    train_y = pd.read_csv('../../../resources/text/train_y.csv', delimiter=',')
    train_x = text_pre_processing('../../../resources/text/train_x.txt')
    test_x = text_pre_processing('../../../resources/text/test.txt')
    test_x = converter(test_x)
    train_x = converter(train_x)

    tf_idf_vectoriser = TfidfVectorizer(min_df=10)
    count_vectoriser = CountVectorizer(min_df=10)

    print("Count vectoriser......")
    abstract_vectoriser(train_x, train_y.values.ravel(), test_x, count_vectoriser)

    print("TF-IDF vectoriser......")
    np.savetxt('../../../resources/text/lab4.csv',
               abstract_vectoriser(train_x, train_y.values.ravel(), test_x, tf_idf_vectoriser), fmt='%d', delimiter=',')


def abstract_vectoriser(train_x, train_y, test_x, vectoriser):
    x_train, x_validate, y_train, y_validate = train_test_split(train_x, train_y,
                                                                test_size=0.25, stratify=train_y, random_state=2)

    x_train_vectoriser = vectoriser.fit_transform(x_train)
    x_validate_vectoriser = vectoriser.transform(x_validate)

    clf = LogisticRegression(penalty='l2')
    forest_clf = RandomForestClassifier(n_estimators=100)
    busting_clf = GradientBoostingClassifier()

    clf.fit(x_train_vectoriser, y_train)
    forest_clf.fit(x_train_vectoriser, y_train)
    busting_clf.fit(x_train_vectoriser, y_train)

    print("Accuracy score logistic regression:",
          metrics.accuracy_score(y_validate, clf.predict(x_validate_vectoriser)))
    print("Accuracy score random forest clf:",
          metrics.accuracy_score(y_validate, forest_clf.predict(x_validate_vectoriser)))
    print("Accuracy score busting clf:",
          metrics.accuracy_score(y_validate, busting_clf.predict(x_validate_vectoriser)))
    return clf.predict(vectoriser.transform(test_x))


def create_x_and_y_train():
    train_x = read_full_file('../../../resources/text/0_train.txt')
    temp_size = len(train_x.split('\n')) - 1
    train_y = filling(temp_size, 0)

    train_x += read_full_file('../../../resources/text/1_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 1)
    temp_size = len(train_x.split('\n'))

    train_x += read_full_file('../../../resources/text/2_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 2)
    temp_size = len(train_x.split('\n'))

    train_x += read_full_file('../../../resources/text/3_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 3)
    temp_size = len(train_x.split('\n'))

    train_x += read_full_file('../../../resources/text/4_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 4)
    temp_size = len(train_x.split('\n'))

    train_x += read_full_file('../../../resources/text/5_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 5)
    temp_size = len(train_x.split('\n'))

    train_x += read_full_file('../../../resources/text/6_train.txt')
    train_y += filling(len(train_x.split('\n')) - temp_size, 6)

    np.savetxt('../../../resources/text/train_y.csv', train_y, fmt='%d', delimiter=',')

    file = open('../../../resources/text/train_x.txt', 'w', encoding='utf-8')
    file.write(train_x)
    file.close()


def text_pre_processing(file_name):
    text = read_file(file_name)
    result_text = []

    for line in text:
        line = utils.convert_number(line.lower())  # преобразование чисел в строку
        line = utils.remove_punctuation(line)  # удаление пунктуации и пробелов
        line = utils.remove_stopwords(line)  # удаление стоп-слов
        line = utils.stem_and_lemmatize_word(line)  # получение корневой формы слова и лематизация
        result_text.append(line)
    return result_text


def converter(array):
    array = [" ".join(i) for i in array]
    return array


def read_file(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    return file.readlines()


def read_full_file(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    return file.read()


def filling(size, number):
    array = []
    for i in range(0, size):
        array.append(number)
    return array


main()

