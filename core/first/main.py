from core.first.textMining import TextMining


tm = TextMining()
b = tm.textAnalyzer("0_train.txt")
print(b.toarray())