from core.second.textMining import TextMining
from core.second.randomForestClassifier import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer


class OntoClassifier():
    def __init__(self):
        """
            Parameters
             ----------
        """
        self.tm = TextMining()
        self.tf_idf = TfidfTransformer()
        self.RFC = RandomForestClassifier()
        self.res = []
        self.names = ["Coding Theory", "Astronomy", "Biology"]
        self.ontoRes = []

    def fit(self, file_path, y):
        """
               Parameters
               ----------
            x : array-like
                Feature dataset.
            y : array-like
                Target values.
        """

        b = self.tm.textAnalyzer(file_path)
        c = self.tf_idf.fit_transform(b)
        self.RFC.fit(c, y)

    def predict(self, file_path):
        """
               Parameters
               ---------
            x : array-like
                data to classification.
        """
        b = self.tm.transform(file_path)
        c = self.tf_idf.transform(b)
        self.res = self.RFC.predict(c)
        return self.res

    def getOnto(self):
        i = 0
        while i < len(self.res):
            k = 0
            print(self.res[i])
            for j in self.res[i]:
                if bool(j == '1') & (self.names[k] not in self.ontoRes):
                    self.ontoRes.append(self.names[k])
                if bool(j == '0') | bool(j == '1'):
                    k = k + 1
            i = i + 1
        return self.ontoRes
