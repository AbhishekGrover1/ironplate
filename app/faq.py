"""Small retrieval-based FAQ for common nutrition and training questions."""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DATA_DIR

THRESHOLD = 0.15


class Faq:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self._questions: list[str] = []
        self._owner: list[int] = []
        for i, entry in enumerate(entries):
            for q in entry["questions"]:
                self._questions.append(q)
                self._owner.append(i)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._matrix = self._vec.fit_transform(self._questions)

    @classmethod
    def load(cls) -> "Faq":
        return cls(json.loads((DATA_DIR / "faq.json").read_text(encoding="utf-8")))

    def lookup(self, text: str) -> Optional[tuple[str, float]]:
        """Return (answer, similarity) when the text is about a known topic.

        A topic pattern decides which entries are eligible; TF-IDF similarity ranks them.
        """
        t = text.lower().replace("’", "'")
        gated = {i for i, entry in enumerate(self.entries) if re.search(entry["when"], t)}
        if not gated:
            return None
        sims = (self._matrix @ self._vec.transform([t]).T).toarray().ravel()
        order = [j for j in np.argsort(-sims) if self._owner[j] in gated]
        if order and sims[order[0]] >= THRESHOLD:
            best = order[0]
            return self.entries[self._owner[best]]["answer"], float(sims[best])
        return None
