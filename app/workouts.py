"""Exercise library and weekly workout generator.

The generator picks a split from the number of training days, fills each day's
movement-pattern slots with exercises that match the available equipment and
the lifter's level, and sets volume from experience and focus. A seeded jitter
makes "another plan" differ while staying valid.
"""
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DATA_DIR
from .foods import norm_phrase

EQUIPMENT_SETS = {
    "gym": {"gym", "dumbbell", "bodyweight", "band"},
    "home": {"dumbbell", "bodyweight", "band"},
    "bodyweight": {"bodyweight"},
}
EQUIPMENT_LABEL = {"gym": "gym", "home": "home with dumbbells", "bodyweight": "no equipment"}
LEVEL_CAP = {"beginner": 2, "intermediate": 3, "advanced": 3}
LEVEL_PREFERRED = {"beginner": 1, "intermediate": 2, "advanced": 3}
MUSCLES = ["chest", "back", "shoulders", "biceps", "triceps", "quads", "hamstrings", "glutes", "calves", "core"]
MUSCLE_WORDS = {
    "chest": ["chest", "pec", "pecs"], "back": ["back", "lat", "lats"],
    "shoulders": ["shoulder", "shoulders", "delt", "delts"], "biceps": ["bicep", "biceps"],
    "triceps": ["tricep", "triceps"], "quads": ["quad", "quads", "quadriceps", "thigh", "thighs"],
    "hamstrings": ["hamstring", "hamstrings"], "glutes": ["glute", "glutes", "butt"],
    "calves": ["calf", "calves"], "core": ["abs", "ab", "core", "stomach", "abdominals"],
}
GROUP_WORDS = {
    "legs": ["quads", "hamstrings", "glutes", "calves"], "leg": ["quads", "hamstrings", "glutes", "calves"],
    "arms": ["biceps", "triceps"], "arm": ["biceps", "triceps"],
}

# Each slot lists acceptable movement patterns in order of preference.
DAY_TEMPLATES = {
    "full_a": [("squat",), ("push_h",), ("pull_h",), ("hinge",), ("lateral", "rear"), ("core",)],
    "full_b": [("hinge", "squat"), ("push_v", "push_h"), ("pull_v", "pull_h"), ("lunge",), ("biceps_iso", "triceps_iso"), ("core",)],
    "full_c": [("lunge", "squat"), ("push_h",), ("pull_v", "pull_h"), ("hinge",), ("lateral", "rear"), ("core",)],
    "push": [("push_h",), ("push_v",), ("push_h", "fly"), ("lateral",), ("triceps_iso",), ("triceps_iso", "core")],
    "pull": [("pull_v",), ("pull_h",), ("rear",), ("biceps_iso",), ("biceps_iso",), ("core",)],
    "legs": [("squat",), ("hinge",), ("lunge", "quad_iso"), ("ham_iso", "hinge"), ("calf",), ("core",)],
    "upper": [("push_h",), ("pull_h",), ("push_v",), ("pull_v",), ("biceps_iso",), ("triceps_iso",)],
    "lower": [("squat",), ("hinge",), ("lunge",), ("ham_iso", "quad_iso", "hinge"), ("calf",), ("core",)],
}
NAMES = {"full_a": "Full body A", "full_b": "Full body B", "full_c": "Full body C", "push": "Push",
         "pull": "Pull", "legs": "Legs", "upper": "Upper", "lower": "Lower"}
SPLITS = {
    1: ("1-day full body", ["full_a"]),
    2: ("2-day full body", ["full_a", "full_b"]),
    "3b": ("3-day full body", ["full_a", "full_b", "full_c"]),
    3: ("3-day push / pull / legs", ["push", "pull", "legs"]),
    4: ("4-day upper / lower", ["upper", "lower", "upper", "lower"]),
    5: ("5-day push / pull / legs + upper / lower", ["push", "pull", "legs", "upper", "lower"]),
    6: ("6-day push / pull / legs", ["push", "pull", "legs", "push", "pull", "legs"]),
}
MAIN_SLOTS = 2  # the first slots of a day are the heavy compound lifts
STAPLES = {
    "barbell back squat", "goblet squat", "bodyweight squat", "romanian deadlift",
    "dumbbell romanian deadlift", "barbell bench press", "dumbbell bench press", "push-up",
    "overhead press", "dumbbell shoulder press", "barbell row", "one-arm dumbbell row",
    "lat pulldown", "pull-up", "walking lunge", "hip thrust", "lateral raise", "dumbbell curl",
    "triceps pushdown", "plank", "leg press", "seated cable row",
}


@dataclass(frozen=True)
class Exercise:
    id: int
    name: str
    aliases: tuple[str, ...]
    pattern: str
    primary: str
    secondary: tuple[str, ...]
    equipment: str
    level: int
    compound: bool
    cue: str

    @property
    def label(self) -> str:
        return self.name[:1].upper() + self.name[1:]


class ExerciseDB:
    def __init__(self, items: list[Exercise]):
        self.items = items
        phrases, owners = [], []
        self._exact: dict[str, Exercise] = {}
        for ex in items:
            for text in (ex.name, *ex.aliases):
                p = norm_phrase(text)
                if p and p not in self._exact:
                    self._exact[p] = ex
                    phrases.append(p)
                    owners.append(ex.id)
        self._owners = np.array(owners)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self._matrix = self._vec.fit_transform(phrases)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ExerciseDB":
        path = path or DATA_DIR / "exercises.csv"
        items = []
        with open(path, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                items.append(Exercise(
                    id=i, name=row["name"].strip().lower(),
                    aliases=tuple(a.strip().lower() for a in row["aliases"].split("|") if a.strip()),
                    pattern=row["pattern"], primary=row["primary"],
                    secondary=tuple(s for s in row["secondary"].split("|") if s),
                    equipment=row["equipment"], level=int(row["level"]),
                    compound=row["compound"] == "1", cue=row["cue"].strip(),
                ))
        return cls(items)

    def search(self, query: str, threshold: float = 0.5) -> Optional[tuple[Exercise, float]]:
        """Best exercise for a name or alias, with a 0-1 confidence."""
        qn = norm_phrase(query)
        if not qn:
            return None
        if qn in self._exact:
            return self._exact[qn], 1.0
        padded = f" {qn} "
        best, best_score = None, 0.0
        for phrase, ex in self._exact.items():
            if len(phrase) >= 4 and f" {phrase} " in padded:
                score = 0.6 + 0.4 * len(phrase) / len(qn)
                if score > best_score:
                    best, best_score = ex, score
        sims = (self._matrix @ self._vec.transform([qn]).T).toarray().ravel()
        j = int(np.argmax(sims))
        if sims[j] > best_score:
            best, best_score = self.items[int(self._owners[j])], float(sims[j])
        return (best, best_score) if best is not None and best_score >= threshold else None

    @staticmethod
    def muscles_in(text: str) -> list[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-z]+", lowered))
        found = [m for m, keys in MUSCLE_WORDS.items() if words & set(keys)]
        if "lower" in words:
            found.extend(m for m in ("quads", "hamstrings", "glutes", "calves") if m not in found)
        if "upper" in words:
            found.extend(m for m in ("chest", "back", "shoulders") if m not in found)
        for group, members in GROUP_WORDS.items():
            if group in words:
                found.extend(m for m in members if m not in found)
        return found

    def for_muscle(self, muscle: str, equipment: str = "gym", limit: int = 6) -> list[Exercise]:
        allowed = EQUIPMENT_SETS[equipment]
        hits = [e for e in self.items if e.primary == muscle and e.equipment in allowed]
        hits.sort(key=lambda e: (-int(e.compound), e.level, e.name))
        return hits[:limit]

    def alternatives(self, ex: Exercise, equipment: str = "gym", limit: int = 4) -> list[Exercise]:
        allowed = EQUIPMENT_SETS[equipment]
        pool = [e for e in self.items if e.id != ex.id and e.pattern == ex.pattern and e.equipment in allowed]
        pool.sort(key=lambda e: (abs(e.level - ex.level), e.name))
        return pool[:limit]


def _prescription(ex: Exercise, slot_index: int, focus: str, level: str) -> dict:
    """Sets, reps and rest for one exercise."""
    if ex.pattern == "cardio":
        reps = {"brisk walk": "20-40 min", "jump rope": "5 x 1 min", "bike intervals": "10-15 min",
                "burpees": "4 x 30 s"}.get(ex.name, "10-20 min")
        return {"sets": 1, "reps": reps, "rest": "as needed"}
    if ex.pattern == "core":
        if "plank" in ex.name:
            return {"sets": 3, "reps": "30-60 s", "rest": "45 s"}
        if ex.name == "mountain climbers":
            return {"sets": 3, "reps": "30 s", "rest": "45 s"}
        if ex.name in {"dead bug", "bird dog"}:
            return {"sets": 3, "reps": "8-10 per side", "rest": "45 s"}
        return {"sets": 3, "reps": "10-15", "rest": "60 s"}
    main = ex.compound and slot_index < MAIN_SLOTS + 1
    if main:
        sets = {"beginner": 3, "intermediate": 4, "advanced": 4}[level]
        reps = {"strength": "3-6", "hypertrophy": "6-10", "general": "8-12"}[focus]
        rest = {"strength": "2-3 min", "hypertrophy": "2 min", "general": "90 s"}[focus]
        if level == "beginner" and focus != "general":
            reps = "6-10" if focus == "strength" else "8-12"
        return {"sets": sets, "reps": reps, "rest": rest}
    sets = {"beginner": 2, "intermediate": 3, "advanced": 3}[level]
    reps = {"strength": "8-12", "hypertrophy": "10-15", "general": "10-15"}[focus]
    return {"sets": sets, "reps": reps, "rest": "60-90 s"}


def build_plan(db: ExerciseDB, days: int = 3, equipment: str = "gym", level: str = "beginner",
               goal: Optional[str] = None, focus: Optional[str] = None, seed: int = 0) -> dict:
    """Generate a weekly plan. Slots that no exercise can fill are skipped."""
    days = max(1, min(int(days), 6))
    equipment = equipment if equipment in EQUIPMENT_SETS else "gym"
    level = level if level in LEVEL_CAP else "beginner"
    focus = focus or ("hypertrophy" if goal == "gain" else "general" if goal == "lose" else "hypertrophy")
    rng = random.Random(seed)
    key = "3b" if days == 3 and level == "beginner" else days
    split_name, template_keys = SPLITS[key]
    allowed = EQUIPMENT_SETS[equipment]
    used_week: dict[str, int] = {}
    plan_days = []
    weekly: dict[str, float] = {m: 0.0 for m in MUSCLES}

    for d, tkey in enumerate(template_keys):
        chosen_today: set[int] = set()
        out = []
        for s, patterns in enumerate(DAY_TEMPLATES[tkey]):
            pick: Optional[Exercise] = None
            for pattern in patterns:
                pool = [e for e in db.items if e.pattern == pattern and e.equipment in allowed
                        and e.level <= LEVEL_CAP[level] and e.id not in chosen_today]
                if not pool:
                    continue

                def score(e: Exercise) -> float:
                    value = rng.uniform(0, 0.6)
                    value -= 0.5 * abs(e.level - LEVEL_PREFERRED[level])
                    value += 1.0 if (e.compound and s < MAIN_SLOTS + 1) else 0.0
                    value -= 0.8 * used_week.get(e.name, 0)
                    if e.name in STAPLES:
                        value += 0.4
                    if equipment == "gym" and e.equipment in {"gym", "dumbbell"}:
                        value += 0.5
                    elif equipment == "home" and e.equipment == "dumbbell":
                        value += 0.4
                    return value

                pick = max(pool, key=score)
                break
            if pick is None:
                continue
            chosen_today.add(pick.id)
            used_week[pick.name] = used_week.get(pick.name, 0) + 1
            rx = _prescription(pick, s, focus, level)
            out.append({"name": pick.label, "muscle": pick.primary, "cue": pick.cue, **rx})
            if pick.pattern != "cardio":
                weekly[pick.primary] = weekly.get(pick.primary, 0) + rx["sets"]
                for sec in pick.secondary:
                    if sec in weekly:
                        weekly[sec] += rx["sets"] * 0.5
        plan_days.append({"name": f"Day {d + 1}: {NAMES[tkey]}", "exercises": out})

    notes = [
        "Warm up with one or two light sets of your first exercise.",
        "Stop each set with one to three reps left in reserve.",
        "When you reach the top of the rep range on every set, add the smallest weight jump.",
    ]
    if goal == "lose":
        notes.append("Add two or three brisk 30-minute walks on rest days and keep daily steps high.")
    if equipment == "bodyweight":
        notes.append("To progress without weights, slow the lowering phase or move to a harder variation.")
    return {
        "title": split_name,
        "summary": f"{EQUIPMENT_LABEL[equipment]}, {level}, {'strength' if focus == 'strength' else 'muscle' if focus == 'hypertrophy' else 'general fitness'} focus",
        "days": plan_days,
        "weekly_sets": {m: round(v) for m, v in weekly.items() if v >= 1},
        "notes": notes,
    }
