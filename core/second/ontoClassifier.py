import numpy as np
from textMining import TextMining
from randomForestClassifier import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer


class OntoClassifier():
    def __init__(self):
        """
            Parameters
             ----------

        """
        self.RFC = RandomForestClassifier()

    def fit(self, file_path, y):
        """
               Parameters
               ----------
            x : array-like
                Feature dataset.
            y : array-like
                Target values.
        """
        tm = TextMining()
        b = tm.textAnalyzer(file_path)
        print(b.toarray())
        tf_idf = TfidfTransformer()
        c = tf_idf.fit_transform(b)
        self.RFC.fit(c, y)

    def predict(self, file_path):
        """
               Parameters
               ---------
            x : array-like
                data to classification.
        """
        tm = TextMining()
        b = tm.textAnalyzer(file_path)
        print(b.toarray())
        tf_idf = TfidfTransformer()
        c = tf_idf.fit_transform(b)
        return self.RFC.predict(c)
