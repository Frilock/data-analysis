from core.first.textMining import TextMining
from nltk.stem.snowball import SnowballStemmer


tm = TextMining()
b = tm.textAnalyzer("0_train.txt")
print(b.toarray())