"""Indonesian NLP preprocessing pipeline for AGRO-SEARCH.
4-stage: case folding → cleaning → stopword removal → stemming (Sastrawi).
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_stemmer = StemmerFactory().createStemmer()
_stopword_remover = StopWordRemoverFactory().createStopWordRemover()


def case_fold(text: str) -> str:
    return text.lower()


def clean(text: str) -> str:
    """Remove punctuation, numbers, extra whitespace."""
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    return _stopword_remover.remove(text)


def stem(text: str) -> str:
    return _stemmer.stem(text)


def preprocess(text: str) -> str:
    """Full 4-stage pipeline: case fold → clean → stopword → stem."""
    return stem(remove_stopwords(clean(case_fold(text))))
