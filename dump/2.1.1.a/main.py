from Vectorizer import CountVectorizer
import pandas as pd
from TextMining import TM

data_0 = pd.read_csv('0_train.txt', delimiter="\t", header=None)
data_0 = data_0.values

data = []
label = []
test = []

i = 0
while i in range(data_0.size):
    data.append(str(data_0[i]))
    label.append(0)
    i += 1

vect = CountVectorizer(ngram_range=(2, 3), analyzer='word')

a = vect.fit_transform(data)
tm = TM()
b = tm.textAnalyze("0_train.txt")