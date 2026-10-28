"""Exercise names and aliases, used to fill training templates."""
from __future__ import annotations

from .workouts import ExerciseDB


def exercise_terms() -> list[str]:
    terms: set[str] = set()
    for ex in ExerciseDB.load().items:
        terms.add(ex.name)
        terms.update(ex.aliases)
    return sorted(terms)
