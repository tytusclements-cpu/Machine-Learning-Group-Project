from collections.abc import Callable
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from spam_detection.config import RANDOM_SEED

WORD_TFIDF = "TF-IDF word 1-2 grams"
WORD_CHAR_TFIDF = "TF-IDF word 1-2 grams + char 3-5 grams"


@dataclass(frozen=True)
class Candidate:
    name: str
    representation: str
    hyperparameters: str
    build: Callable[[], Pipeline]


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)


def word_and_char_tfidf() -> FeatureUnion:
    return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])


def naive_bayes(alpha: float) -> Pipeline:
    return make_pipeline(word_tfidf(), MultinomialNB(alpha=alpha))


def logistic_regression(c: float) -> Pipeline:
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    return make_pipeline(word_tfidf(), classifier)


def linear_svm(features: Callable, c: float) -> Pipeline:
    classifier = LinearSVC(C=c, class_weight="balanced", random_state=RANDOM_SEED)
    return make_pipeline(features(), classifier)


CANDIDATES = [
    Candidate("naive_bayes_baseline", WORD_TFIDF, "alpha=1.0", lambda: naive_bayes(1.0)),
    Candidate("naive_bayes_low_alpha", WORD_TFIDF, "alpha=0.1", lambda: naive_bayes(0.1)),
    Candidate(
        "logistic_regression",
        WORD_TFIDF,
        "C=10, class_weight=balanced",
        lambda: logistic_regression(10),
    ),
    Candidate(
        "linear_svm",
        WORD_TFIDF,
        "C=1, class_weight=balanced",
        lambda: linear_svm(word_tfidf, 1),
    ),
    Candidate(
        "linear_svm_word_char",
        WORD_CHAR_TFIDF,
        "C=1, class_weight=balanced",
        lambda: linear_svm(word_and_char_tfidf, 1),
    ),
]
