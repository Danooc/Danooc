"""
Text preprocessing for Spanish call center speech input.
"""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_STOPWORDS_ES = frozenset(
    "de la el en y a los del las un una unos unas por con para al es lo que "
    "se le su sus me mi nos te tu tus".split()
)


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in _STOPWORDS_ES)


def preprocess(text: str) -> str:
    return remove_stopwords(normalize(text))
