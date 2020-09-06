from core.first.vectorizer import CountVectorizer
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


class TextMining(object):
    def __init__(self):  #  текст, разбитый на предложения
        """
            Parameters
                ----------
        """
        self.engine = inflect.engine()
        self.stemmer = SnowballStemmer("russian")  # замена на русский стеммер

    def pre_processing(self, file_path, encoding):
        text = self.open_file(file_path, encoding)
        text = self.split_text_in_sentences(text)
        result_text = []

        for suggestion in text:
            suggestion = suggestion.lower()
            suggestion = self.convert_number(suggestion)
            suggestion = self.remove_punctuation(suggestion)
            suggestion = self.remove_stopwords(suggestion)  # добавлена токенизация
            suggestion = self.stem_and_lemmatize_word(suggestion)  # вариативно
            result_text.append(suggestion)
        return self.converter(result_text)

    def split_text_in_sentences(self, text):
        # Проблема в том, что точка часто не явл. разделителем предложения.
        # И (иногда!) восклицательный знак тоже.
        # Обязательно надо проверять регистр буквы после точки.
        text_in_sentences = re.split(r'(?<=[.!?…])', text)
        text_in_sentences.remove(text_in_sentences[-1])  # удаление последнего пустого предложения
        return text_in_sentences

    def converter(self, text):
        text = [" ".join(i) for i in text]
        return text

    def convert_number(self, text):
        temp_str = text.split()
        new_string = []
        for word in temp_str:
            if word.isdigit():
                temp = self.engine.number_to_words(word)
                new_string.append(temp)
            else:
                new_string.append(word)
        temp_str = ' '.join(new_string)
        return temp_str

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return " ".join(text.split())

    def remove_stopwords(self, text):
        stop_words = stopwords.words('russian')
        word_tokens = word_tokenize(text, language="russian")
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return filtered_text

    def stem_and_lemmatize_word(self, text):
        word_tokens = [self.stemmer.stem(word) for word in text]  # посмотреть результаты, возможно 1 убрать
        return word_tokens

    def split_array(self, array):
        array_split = []
        for i in array:
            array_split.append(i.split())
        return array_split

    def open_file(self, file_path, encoding):
        file = open(file_path, 'r', encoding=encoding)
        return file.read()

    def textAnalyzer(self,file_path):
        text = self.pre_processing(file_path, 'utf-8')
        vect = CountVectorizer(ngram_range=(2, 3), analyzer='word')
        result = vect.fit_transform(text)
        return result

