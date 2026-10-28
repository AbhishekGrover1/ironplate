"""Energy and macro targets: Mifflin-St Jeor, activity factor, goal adjustment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
ACTIVITY_LABEL = {
    "sedentary": "sedentary", "light": "lightly active", "moderate": "moderately active",
    "active": "very active", "very_active": "extra active",
}
SEX_CONSTANT = {"male": 5, "female": -161, "other": -78}
CALORIE_FLOOR = {"male": 1500, "female": 1200, "other": 1350}
PACE_LOSE = {"gentle": 0.10, "steady": 0.15, "fast": 0.20}
PACE_GAIN = {"gentle": 0.05, "steady": 0.10, "fast": 0.15}
KCAL_PER_KG = 7700.0
REQUIRED = ("age", "sex", "height_cm", "weight_kg")


@dataclass
class Targets:
    bmr: int
    tdee: int
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    fiber_g: int
    bmi: float
    bmi_label: str
    weekly_change_kg: float
    goal: str
    activity: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def missing_fields(profile: dict) -> list[str]:
    return [f for f in REQUIRED if profile.get(f) in (None, "")]


def bmi_label(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "healthy range"
    if bmi < 30:
        return "overweight"
    return "obese range"


def activity_from_days(days: Optional[int]) -> Optional[str]:
    if days is None:
        return None
    if days <= 0:
        return "sedentary"
    if days <= 3:
        return "light"
    if days <= 5:
        return "moderate"
    return "active"


def compute_targets(profile: dict) -> Targets:
    """Daily calories and macros from a complete profile. Raises KeyError if incomplete."""
    sex, age = profile["sex"], float(profile["age"])
    height, weight = float(profile["height_cm"]), float(profile["weight_kg"])
    goal = profile.get("goal") or "maintain"
    activity = profile.get("activity") or activity_from_days(profile.get("days")) or "light"
    pace = profile.get("pace") or "steady"
    notes: list[str] = []

    bmr = 10 * weight + 6.25 * height - 5 * age + SEX_CONSTANT[sex]
    tdee = bmr * ACTIVITY[activity]
    bmi = weight / (height / 100) ** 2

    effective = goal
    if goal == "lose" and bmi < 18.5:
        effective = "maintain"
        notes.append("Your BMI is in the underweight range, so I kept calories at maintenance instead of a deficit.")

    if effective == "lose":
        deficit = min(tdee * PACE_LOSE[pace], 0.01 * weight * KCAL_PER_KG / 7)
        calories = tdee - deficit
    elif effective == "gain":
        calories = tdee * (1 + PACE_GAIN[pace])
    else:
        calories = tdee

    floor = min(CALORIE_FLOOR[sex], tdee)
    if calories < floor:
        calories = floor
        notes.append(f"Calories are held at {round(floor)}, a minimum I do not go below.")
    calories = round(calories / 10) * 10

    ideal25 = 25 * (height / 100) ** 2
    ref_w = weight if bmi < 30 else ideal25 + 0.5 * (weight - ideal25)
    per_kg = {"lose": 2.0 if pace != "fast" else 2.2, "maintain": 1.6, "gain": 1.8}[effective]
    protein = min(per_kg * ref_w, 0.35 * calories / 4)
    fat = min(max(0.25 * calories / 9, 0.6 * ref_w), 0.40 * calories / 9)
    carbs = (calories - 4 * protein - 9 * fat) / 4
    if carbs < 80:
        protein = max(1.2 * ref_w, (calories - 9 * fat - 4 * 80) / 4)
        carbs = (calories - 4 * protein - 9 * fat) / 4
    carbs = max(carbs, 0.0)
    fiber = min(max(round(14 * calories / 1000), 20), 40)

    notes.append("Estimates are within about 10 percent. Check your weight trend after two to three weeks and adjust.")
    return Targets(
        bmr=round(bmr), tdee=round(tdee / 10) * 10, calories=int(calories),
        protein_g=round(protein), carbs_g=round(carbs), fat_g=round(fat), fiber_g=int(fiber),
        bmi=round(bmi, 1), bmi_label=bmi_label(bmi),
        weekly_change_kg=round((calories - tdee) * 7 / KCAL_PER_KG, 2),
        goal=effective, activity=activity, notes=notes,
    )
