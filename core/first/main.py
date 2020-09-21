from core.first.textMining import TextMining
tm = TextMining()
b = tm.fit_transform("0_train.txt")
print(b.toarray())
