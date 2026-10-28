"""Intent classifier: word + character n-gram TF-IDF into a logistic regression."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODEL_DIR

MODEL_FILE = "intent_model.joblib"
CONFIDENCE_MIN = 0.35


def classifier_text(text: str) -> str:
    """Normalise a message for the classifier: numbers become a token, punctuation goes."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\d+(?:\.\d+)?\s*'\s*\d*\"?", " numft ", t)
    t = re.sub(r"\d+(?:[.,]\d+)?", " num ", t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=classifier_text, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=classifier_text, analyzer="char_wb",
                                 ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=30.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class IntentModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.classes: list[str] = list(pipeline.classes_)

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        probs = self.pipeline.predict_proba([text])[0]
        best = int(np.argmax(probs))
        return self.classes[best], float(probs[best]), {c: float(p) for c, p in zip(self.classes, probs)}

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = directory or MODEL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILE
        joblib.dump(self.pipeline, path, compress=3)
        return path

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "IntentModel":
        return cls(joblib.load((directory or MODEL_DIR) / MODEL_FILE))
