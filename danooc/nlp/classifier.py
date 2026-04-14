"""
Trainable intent classifier for call center speech input.

Uses TF-IDF vectorisation + a simple linear model (logistic regression)
trained from labeled examples. The entire model serialises to a single
JSON file — no pickle, no heavy ML frameworks needed at runtime.
"""
from __future__ import annotations

import json
import math
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from danooc.nlp.preprocessing import preprocess

logger = logging.getLogger(__name__)


@dataclass
class _VocabEntry:
    index: int
    idf: float = 0.0


class IntentClassifier:
    """
    Lightweight intent classifier.

    Training algorithm:
      1. Build vocabulary + IDF weights from labelled examples.
      2. For each class, compute a centroid (mean TF-IDF vector).
      3. At inference, compute cosine similarity to each centroid.

    This is equivalent to a nearest-centroid classifier in TF-IDF space
    and works surprisingly well for small intent datasets (< 500 examples).
    """

    def __init__(self) -> None:
        self._vocab: Dict[str, _VocabEntry] = {}
        self._centroids: Dict[str, Dict[int, float]] = {}
        self._labels: List[str] = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, examples: List[Dict[str, str]]) -> dict:
        """
        Train from a list of ``{"text": ..., "intent": ...}`` dicts.

        Returns training metrics.
        """
        if not examples:
            raise ValueError("Need at least one training example")

        texts = [preprocess(e["text"]) for e in examples]
        labels = [e["intent"] for e in examples]
        self._labels = sorted(set(labels))

        self._build_vocab(texts)
        vectors = [self._tfidf(t) for t in texts]

        groups: Dict[str, List[Dict[int, float]]] = defaultdict(list)
        for vec, lbl in zip(vectors, labels):
            groups[lbl].append(vec)

        self._centroids = {}
        for lbl, vecs in groups.items():
            centroid: Dict[int, float] = defaultdict(float)
            for v in vecs:
                for idx, val in v.items():
                    centroid[idx] += val
            n = len(vecs)
            self._centroids[lbl] = {idx: val / n for idx, val in centroid.items()}

        self._trained = True

        correct = sum(1 for t, l in zip(texts, labels) if self.predict(t)[0] == l)
        accuracy = correct / len(texts) if texts else 0
        metrics = {
            "num_examples": len(texts),
            "num_intents": len(self._labels),
            "vocab_size": len(self._vocab),
            "train_accuracy": round(accuracy * 100, 2),
        }
        logger.info("Training complete: %s", metrics)
        return metrics

    def _build_vocab(self, texts: List[str]) -> None:
        doc_freq: Counter = Counter()
        all_words: set = set()
        for text in texts:
            words = set(text.split())
            doc_freq.update(words)
            all_words.update(words)

        n_docs = len(texts)
        self._vocab = {}
        for i, word in enumerate(sorted(all_words)):
            idf = math.log((1 + n_docs) / (1 + doc_freq[word])) + 1
            self._vocab[word] = _VocabEntry(index=i, idf=idf)

    def _tfidf(self, text: str) -> Dict[int, float]:
        words = text.split()
        tf = Counter(words)
        total = len(words) if words else 1
        vec: Dict[int, float] = {}
        for w, count in tf.items():
            entry = self._vocab.get(w)
            if entry:
                vec[entry.index] = (count / total) * entry.idf
        return vec

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Classify ``text`` and return ``(intent, confidence)``.
        """
        if not self._trained:
            raise RuntimeError("Classifier not trained — call train() or load() first")

        processed = preprocess(text)
        vec = self._tfidf(processed)

        best_label = "unknown"
        best_score = -1.0
        for lbl, centroid in self._centroids.items():
            score = self._cosine(vec, centroid)
            if score > best_score:
                best_score = score
                best_label = lbl

        confidence = max(0.0, min(1.0, best_score))
        return best_label, round(confidence, 4)

    def predict_top_n(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """Return top *n* intents with confidence scores."""
        if not self._trained:
            raise RuntimeError("Classifier not trained")
        processed = preprocess(text)
        vec = self._tfidf(processed)
        scores = []
        for lbl, centroid in self._centroids.items():
            scores.append((lbl, round(max(0.0, self._cosine(vec, centroid)), 4)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    @staticmethod
    def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        vocab_data = {w: {"i": e.index, "idf": e.idf} for w, e in self._vocab.items()}
        centroids_data = {lbl: {str(k): v for k, v in c.items()} for lbl, c in self._centroids.items()}
        data = {
            "vocab": vocab_data,
            "centroids": centroids_data,
            "labels": self._labels,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._vocab = {
            w: _VocabEntry(index=e["i"], idf=e["idf"])
            for w, e in data["vocab"].items()
        }
        self._centroids = {
            lbl: {int(k): v for k, v in c.items()}
            for lbl, c in data["centroids"].items()
        }
        self._labels = data["labels"]
        self._trained = True
        logger.info("Model loaded from %s (%d intents, %d vocab)", path, len(self._labels), len(self._vocab))
