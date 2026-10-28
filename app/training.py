"""Build the intent dataset, train the classifier and write evaluation metrics.

Phrases are authored by hand and expanded with templates that are filled with
real food names, exercises and numbers. Cross-validation is grouped so that every
variant of one phrase or template stays in the same fold; the reported scores
therefore describe unseen phrasings rather than memorised ones.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from .config import DATA_DIR, MODEL_DIR
from .exercises_terms import exercise_terms
from .foods import FoodDB
from .nlu import CONFIDENCE_MIN, IntentModel, make_pipeline

PER_TEMPLATE = 16
PREFIXES = ["please ", "can you ", "hey ", "i want to ", "could you ", "i need to "]
SUFFIXES = [" please", " thanks", " now", " for me"]
NO_AFFIX = {"greet", "thanks", "bye", "log_food", "log_weight", "set_profile", "out_of_scope"}


def food_terms(db: FoodDB) -> list[str]:
    terms: set[str] = set()
    for f in db.foods:
        terms.add(f.name.split(" (")[0])
        terms.update(f.aliases)
    return sorted(terms)


def _weight(rng: random.Random) -> str:
    kg = round(rng.uniform(45, 125), rng.choice([0, 1]))
    kg = int(kg) if kg == int(kg) else kg
    form = rng.choice(["{kg} kg", "{kg}kg", "{lb} lbs", "{kg}", "{kg} kgs"])
    return form.format(kg=kg, lb=round(kg * 2.2))


def _height(rng: random.Random) -> str:
    cm = rng.randint(150, 195)
    feet, inch = divmod(round(cm / 2.54), 12)
    form = rng.choice(["{cm} cm", "{cm}cm", "{ft}'{inch}", "{ft} ft {inch} in", "{m:.2f} m"])
    return form.format(cm=cm, ft=feet, inch=inch, m=cm / 100)


def build_dataset(seed: int = 7, per_template: int = PER_TEMPLATE) -> tuple[list[str], list[str], list[str]]:
    """Return (texts, labels, groups)."""
    rng = random.Random(seed)
    spec = json.loads((DATA_DIR / "intents.json").read_text(encoding="utf-8"))
    slots = spec["slots"]
    foods = food_terms(FoodDB.load())
    exercises = exercise_terms()
    dynamic = {
        "food": lambda: rng.choice(foods), "exercise": lambda: rng.choice(exercises),
        "weight": lambda: _weight(rng), "height": lambda: _height(rng),
        "age": lambda: str(rng.randint(16, 70)),
        "sex": lambda: rng.choice(["male", "female", "man", "woman", "guy", "girl"]),
    }

    def fill(template: str) -> str:
        out = template
        while "{" in out:
            start = out.index("{")
            end = out.index("}", start)
            name = out[start + 1:end]
            value = dynamic[name]() if name in dynamic else rng.choice(slots[name])
            out = out[:start] + value + out[end + 1:]
        return out

    texts, labels, groups = [], [], []
    for intent, block in spec["intents"].items():
        for i, example in enumerate(block["examples"]):
            gid = f"{intent}:e{i}"
            variants = [example]
            if intent not in NO_AFFIX:
                variants.append(rng.choice(PREFIXES) + example)
                variants.append(example + rng.choice(SUFFIXES))
            for v in variants:
                texts.append(v)
                labels.append(intent)
                groups.append(gid)
        for j, template in enumerate(block["templates"]):
            seen: set[str] = set()
            for _ in range(per_template * 3):
                text = fill(template)
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
                    labels.append(intent)
                    groups.append(f"{intent}:t{j}")
                if len(seen) >= per_template:
                    break
    return texts, labels, groups


def evaluate(texts: list[str], labels: list[str], groups: list[str], folds: int = 5) -> dict:
    pipeline = make_pipeline()
    probs = cross_val_predict(pipeline, texts, labels, groups=groups, cv=GroupKFold(n_splits=folds),
                              method="predict_proba")
    classes = sorted(set(labels))
    pred = np.array(classes)[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    y = np.array(labels)
    keep = conf >= CONFIDENCE_MIN
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return {
        "folds": folds,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "confidence_threshold": CONFIDENCE_MIN,
        "answered_share": round(float(keep.mean()), 4),
        "accuracy_when_answered": round(float(accuracy_score(y[keep], pred[keep])), 4) if keep.any() else None,
        "per_intent": {c: {"f1": round(report[c]["f1-score"], 3), "support": int(report[c]["support"])} for c in classes},
    }


def train(seed: int = 7, write: bool = True, folds: int = 5) -> dict:
    texts, labels, groups = build_dataset(seed)
    metrics = evaluate(texts, labels, groups, folds)
    pipeline = make_pipeline().fit(texts, labels)
    model = IntentModel(pipeline)
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": len(texts), "groups": len(set(groups)), "intents": len(set(labels)),
        "scikit_learn": sklearn.__version__,
        "cross_validation": metrics,
        "note": ("Grouped cross-validation on authored phrases and templates. Scores describe unseen "
                 "phrasings of the same kind, not open-domain text."),
    }
    if write:
        model.save()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    summary = train()
    cv = summary["cross_validation"]
    print(f"Trained on {summary['samples']} samples, {summary['intents']} intents.")
    print(f"Grouped {cv['folds']}-fold CV: accuracy {cv['accuracy']:.3f}, macro F1 {cv['macro_f1']:.3f}")
    weakest = sorted(cv["per_intent"].items(), key=lambda kv: kv[1]["f1"])[:3]
    print("Weakest intents:", ", ".join(f"{k} ({v['f1']:.2f})" for k, v in weakest))
    print(f"Model written to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def ensure_model() -> IntentModel:
    """Load the trained model, training it first if it is missing or built with another scikit-learn."""
    metrics = MODEL_DIR / "metrics.json"
    try:
        info = json.loads(metrics.read_text(encoding="utf-8"))
        if info.get("scikit_learn") != sklearn.__version__:
            raise ValueError("scikit-learn version changed")
        return IntentModel.load()
    except Exception:
        train()
        return IntentModel.load()
