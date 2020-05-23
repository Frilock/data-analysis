import numpy as np
import pandas as pd
import python.classifier.text.textUtils as utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from gensim.models import Word2Vec


def main():
    # create_x_and_y_train()
    train_y = pd.read_csv('../../../resources/text/train_y.csv', delimiter=',')
    train_x = text_pre_processing('../../../resources/text/train_x.txt')
    test_x = text_pre_processing('../../../resources/text/test.txt')

    tf_idf_vectoriser = TfidfVectorizer(min_df=5)
    count_vectoriser = CountVectorizer(min_df=5)

    print("Word2Vec vectoriser......")
    np.savetxt('../../../resources/text/lab4.csv',
               word_vectoriser(train_x, train_y.values.ravel(), test_x), fmt='%d', delimiter=',')

    print("Count vectoriser......")
    abstract_vectoriser(train_x, train_y.values.ravel(), test_x, count_vectoriser)

    print("TF-IDF vectoriser......")
    abstract_vectoriser(train_x, train_y.values.ravel(), test_x, tf_idf_vectoriser)


def word_vectoriser(tokens_train, train_y, tokens_test):
    size = 300

    w2v_train = utils.split_array(tokens_train)

    model = Word2Vec(w2v_train, size=size, min_count=5, iter=75)

    dictionary = dict(zip(model.wv.index2word, model.wv.vectors))

    x_for_w2v_train = prepare_w2v(size, dictionary, tokens_train)
    x_for_w2v_test = prepare_w2v(size, dictionary, tokens_test)

    x_train, x_validate, y_train, y_validate = \
        train_test_split(x_for_w2v_train, train_y, test_size=0.25, stratify=train_y, random_state=2)

    clf = LogisticRegression(penalty='l2').fit(x_train, y_train)
    print("Accuracy score logistic regression:",
          metrics.accuracy_score(y_validate, clf.predict(x_validate)))
    return clf.predict(x_for_w2v_test)


def abstract_vectoriser(train_x, train_y, test_x, vectoriser):
    x_train, x_validate, y_train, y_validate = train_test_split(train_x, train_y,
                                                                test_size=0.25, stratify=train_y, random_state=2)
    x_train_vectoriser = vectoriser.fit_transform(x_train)
    x_validate_vectoriser = vectoriser.transform(x_validate)

    clf = LogisticRegression(penalty='l2').fit(x_train_vectoriser, y_train)

    print("Accuracy score logistic regression:",
          metrics.accuracy_score(y_validate, clf.predict(x_validate_vectoriser)))
    return clf.predict(vectoriser.transform(test_x))


def prepare_w2v(size, dictionary, tokens):
    x_for_w2v = []
    for token in tokens:
        x_vector = np.zeros(size)
        for word in token.split():
            if word in dictionary:
                x_vector += dictionary[word]
        x_for_w2v.append(x_vector / len(token.split()))
    return x_for_w2v


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
    return utils.converter(result_text)


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
