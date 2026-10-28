"""Request and response models, and a sanitiser for the state the browser sends.

The browser keeps the profile, food log and weights in local storage and sends
them with every message, so the server stays stateless. Anything stale, tampered
with or malformed is dropped here instead of reaching the model code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from .foods import DIET_LEVEL, MEALS
from .nutrition import ACTIVITY
from .trend import parse_day
from .workouts import EQUIPMENT_SETS

ENUMS = {
    "sex": {"male", "female", "other"}, "goal": {"lose", "maintain", "gain"},
    "activity": set(ACTIVITY), "diet": set(DIET_LEVEL), "equipment": set(EQUIPMENT_SETS),
    "experience": {"beginner", "intermediate", "advanced"}, "pace": {"gentle", "steady", "fast"},
    "focus": {"strength", "hypertrophy", "general"}, "units": {"metric", "imperial"},
}
NUMBERS = {
    "age": (5, 110, int), "height_cm": (80.0, 250.0, float), "weight_kg": (20.0, 400.0, float),
    "goal_weight_kg": (20.0, 400.0, float), "days": (0, 7, int), "meals": (3, 5, int),
}
ALLERGENS = {"dairy", "gluten", "nuts", "peanut", "soy", "egg", "fish", "shellfish", "sesame"}
PENDING_TYPES = {"need_profile", "confirm_reset"}
MAX_LOG, MAX_SERIES = 120, 400


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=600)
    state: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    blocks: list[dict[str, Any]] = []
    profile: dict[str, Any] = {}
    pending: Optional[dict[str, Any]] = None
    plan_seed: int = 0
    targets: Optional[dict[str, Any]] = None
    actions: list[dict[str, Any]] = []
    chips: list[str] = []
    intent: str = ""
    confidence: float = 0.0


def sanitize_profile(raw: Any) -> dict:
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    for key, allowed in ENUMS.items():
        if raw.get(key) in allowed:
            out[key] = raw[key]
    for key, (lo, hi, cast) in NUMBERS.items():
        try:
            value = cast(float(raw[key]))
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= value <= hi:
            out[key] = round(value, 1) if cast is float else value
    allergens = raw.get("allergens")
    if isinstance(allergens, list):
        out["allergens"] = sorted({a for a in allergens if a in ALLERGENS})
    avoid = raw.get("avoid")
    if isinstance(avoid, list):
        out["avoid"] = sorted({int(a) for a in avoid if isinstance(a, int) and 0 <= a < 10_000})[:20]
    return out


def _num(value: Any, lo: float = 0.0, hi: float = 20_000.0) -> float:
    try:
        return max(lo, min(float(value), hi))
    except (TypeError, ValueError):
        return 0.0


def sanitize_entries(raw: Any) -> list[dict]:
    entries = []
    for item in (raw if isinstance(raw, list) else [])[:MAX_LOG]:
        if not isinstance(item, dict):
            continue
        meal = item.get("meal") if item.get("meal") in MEALS else "snack"
        entries.append({
            "id": str(item.get("id", ""))[:24], "meal": meal, "name": str(item.get("name", ""))[:80],
            "grams": _num(item.get("grams")), "kcal": _num(item.get("kcal")),
            "protein": _num(item.get("protein")), "carbs": _num(item.get("carbs")), "fat": _num(item.get("fat")),
        })
    return entries


def sanitize_series(raw: Any) -> list[tuple[str, float]]:
    out = []
    for item in (raw if isinstance(raw, list) else [])[:MAX_SERIES]:
        if isinstance(item, dict):
            day, value = item.get("date"), item.get("kg", item.get("kcal"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            day, value = item
        else:
            continue
        if parse_day(day) is not None:
            out.append((str(day)[:10], _num(value, 0.0, 10_000.0)))
    return out


@dataclass
class ClientState:
    profile: dict = field(default_factory=dict)
    pending: Optional[dict] = None
    plan_seed: int = 0
    today: str = field(default_factory=lambda: date.today().isoformat())
    hour: Optional[int] = None
    log_today: list[dict] = field(default_factory=list)
    weights: list[tuple[str, float]] = field(default_factory=list)
    intake: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "ClientState":
        raw = raw if isinstance(raw, dict) else {}
        pending = raw.get("pending")
        if not (isinstance(pending, dict) and pending.get("type") in PENDING_TYPES):
            pending = None
        else:
            pending = {k: v for k, v in pending.items() if k in {"type", "then", "missing"}}
        today = raw.get("today")
        hour = raw.get("hour")
        try:
            seed = max(0, min(int(raw.get("plan_seed", 0)), 10_000))
        except (TypeError, ValueError):
            seed = 0
        return cls(
            profile=sanitize_profile(raw.get("profile")), pending=pending, plan_seed=seed,
            today=today if parse_day(today) else date.today().isoformat(),
            hour=hour if isinstance(hour, int) and 0 <= hour <= 23 else None,
            log_today=sanitize_entries(raw.get("log_today")),
            weights=sanitize_series(raw.get("weights")), intake=sanitize_series(raw.get("intake")),
        )
