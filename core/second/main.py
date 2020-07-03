import pandas as pd
from ontoClassifier import OntoClassifier


data_y = pd.read_csv('y_train.txt', delimiter="\t", header=None)
data_y = data_y.values

label = []

i = 0
while i in range(data_y.size):
    label.append(list(data_y[i]))
    i += 1

oc = OntoClassifier()
oc.fit("0_train.txt", label)
res = oc.predict("0_train.txt")
print(res)
