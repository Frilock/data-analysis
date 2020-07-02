import array
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
import numbers
from operator import itemgetter
import re
import unicodedata

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.utils import _IS_32BIT
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _deprecate_positional_args

__all__ = ['CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'strip_accents_unicode']


def _analyze(doc, analyzer=None, tokenizer=None, ngrams=None, decoder=None):
    print("_analyze")
    """Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing.
    If analyzer is used, only the decoder argument is used, as the analyzer is
    intended to replace tokenizer, and ngrams steps.
    Parameters
    ----------
    analyzer: callable, default=None
    tokenizer: callable, default=None
    ngrams: callable, default=None
    preprocessor: callable, default=None
    decoder: callable, default=None
    stop_words: list, default=None
    Returns
    -------
    ngrams: list
        A sequence of tokens, possibly with pairs, triples, etc.
    """

    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            doc = ngrams(doc)
    return doc


def strip_accents_unicode(s):
    print("strip_accents_unicode")
    """Transform accentuated unicode symbols into their simple counterpart
    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.
    Parameters
    ----------
    s : string
        The string to strip
    See Also
    """
    try:
        # If `s` is ASCII-compatible, then it does not contain any accented
        # characters and we can avoid an expensive list comprehension
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize('NFKD', s)
        return ''.join([c for c in normalized if not unicodedata.combining(c)])


class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        print("decode")
        """Decode the input into a string of unicode symbols.
        The decoding strategy depends on the vectorizer parameters.
        Parameters
        ----------
        doc : str
            The string to decode.
        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc

    def _word_ngrams(self, tokens):
        print("_word_ngrams")
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def build_tokenizer(self):
        print("build_tokenizer")
        """Return a function that splits a string into a sequence of tokens.
        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return token_pattern.findall

    def build_analyzer(self):
        print("build_analyzer")
        """Return a callable that handles tokenization
        and n-grams generation.
        Returns
        -------
        analyzer: callable
            A function to handle tokenization
            and n-grams generation.
        """

        if callable(self.analyzer):
            return partial(
                _analyze, analyzer=self.analyzer, decoder=self.decode
            )

        if self.analyzer == 'word':
            tokenize = self.build_tokenizer()
            return partial(_analyze, ngrams=self._word_ngrams,
                           tokenizer=tokenize,
                           decoder=self.decode)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _validate_vocabulary(self):
        print("_validate_vocabulary")
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(vocabulary.values())
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):
        print("_check_vocabulary")
        """Check if vocabulary is empty or missing (not fitted)"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
            if not self.fixed_vocabulary_:
                raise NotFittedError("Vocabulary not fitted or provided")

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    def _validate_params(self):
        print("_validate_params")
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "lower boundary larger than the upper boundary."
                % str(self.ngram_range))


class VectorizerMixin(_VectorizerMixin):
    pass


class CountVectorizer(_VectorizerMixin, BaseEstimator):
    """Convert a collection of text documents to a matrix of token counts
    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.
    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.
    Read more in the :ref:`User Guide <text_feature_extraction>`.
    Parameters
    ----------
    input : string {'filename', 'file', 'content'}, default='content'
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.
    encoding : string, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode'}, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer is not callable``.
    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    stop_words : string {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.
    analyzer : string, {'word', 'char', 'char_wb'} or callable, \
            default='word'
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
        .. versionchanged:: 0.21
        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
        first read from the file and then passed to the given callable
        analyzer.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    fixed_vocabulary_: boolean
        True if a fixed vocabulary of term to indices mapping
        is provided by the user
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.

    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """

    @_deprecate_positional_args
    def __init__(self, *, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def _sort_features(self, X, vocabulary):
        print("_sort_features")
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def _count_vocab(self, raw_documents, fixed_vocab):
        print("_count_vocab")
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(('sparse CSR array has {} non-zero '
                                  'elements and requires 64 bit indexing, '
                                  'which is unsupported with 32 bit Python.')
                                 .format(indptr[-1]))
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X

    def fit_transform(self, raw_documents, y=None):
        print("fit_transform")
        """Learn the vocabulary dictionary and return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")

            X = self._sort_features(X, vocabulary)
            self.vocabulary_ = vocabulary

        return X

    def get_feature_names(self):
        print("get_feature_names")
        """Array mapping from feature integer indices to feature name.
        Returns
        -------
        feature_names : list
            A list of feature names.
        """

        self._check_vocabulary()

        return [t for t, i in sorted(self.vocabulary_.items(),
                                     key=itemgetter(1))]

    def _more_tags(self):
        print("_more_tags")
        return {'X_types': ['string']}


def _make_int_array():
    print("_make_int_array")
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))
