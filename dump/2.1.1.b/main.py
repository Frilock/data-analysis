import pandas as pd
import numpy as np
from TextMining import TM
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

data_0 = pd.read_csv('0_train.txt', delimiter="\t", header=None)
data_0 = data_0.values
data_y = pd.read_csv('y_train.txt', delimiter="\t", header=None)
data_y = data_y.values

data = []
label = []
test = []

i = 0
while i in range(data_0.size):
    data.append(str(data_0[i]))
    label.append(list(data_y[i]))
    i += 1

tm = TM()
b = tm.textAnalyze("0_train.txt")
print(b.toarray())
tf_idf = TfidfTransformer()
c = tf_idf.fit_transform(b)
RFC = RandomForestClassifier()
RFC.fit(c, label)
res = RFC.predict(b)
print(np.mean(res==label))
