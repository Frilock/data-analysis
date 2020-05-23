import inflect
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

engine = inflect.engine()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return " ".join(text.split())


def convert_number(text):
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        if word.isdigit():
            temp = engine.number_to_words(word)
            new_string.append(temp)
        else:
            new_string.append(word)
    temp_str = ' '.join(new_string)
    return temp_str


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


def stem_and_lemmatize_word(word_tokens):
    word_tokens = [stemmer.stem(word) for word in word_tokens]  # посмотреть результаты, возможно 1 убрать
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return lemmas


def converter(array):
    array = [" ".join(i) for i in array]
    return array


def split_array(array):
    array_split = []
    for i in array:
        array_split.append(i.split())
    return array_split
