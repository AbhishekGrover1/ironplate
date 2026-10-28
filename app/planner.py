"""Day meal planner.

Choosing foods and portions that hit calorie and macro targets is a small
constrained optimisation problem, so it is solved as a mixed-integer program
(SciPy / HiGHS). Portions are whole multiples of a realistic step (half a
banana, one roti, a quarter cup of rice). Each meal must have a sensible
structure, no food repeats across meals, and a seeded cost term varies the
result so "another plan" gives a different day.
"""
from __future__ import annotations

import math
import random
import time
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .foods import Food, FoodDB

SLOT_PLANS = {
    3: [("breakfast", "Breakfast", 0.30), ("lunch", "Lunch", 0.38), ("dinner", "Dinner", 0.32)],
    4: [("breakfast", "Breakfast", 0.25), ("lunch", "Lunch", 0.33),
        ("snack", "Snack", 0.12), ("dinner", "Dinner", 0.30)],
    5: [("breakfast", "Breakfast", 0.22), ("snack", "Mid-morning snack", 0.10),
        ("lunch", "Lunch", 0.30), ("snack", "Evening snack", 0.10), ("dinner", "Dinner", 0.28)],
}
ROLE_ORDER = ["protein", "carb", "veg", "dairy", "fruit", "fat", "snack"]

REQUIRED_ROLES = {
    "breakfast": [({"protein", "dairy"}, 1), ({"carb", "fruit"}, 1)],
    "lunch": [({"protein"}, 1), ({"carb"}, 1), ({"veg"}, 1)],
    "dinner": [({"protein"}, 1), ({"carb"}, 1), ({"veg"}, 1)],
    "snack": [],
}
ROLE_CAPS = {
    "breakfast": {"protein": 2, "dairy": 1, "carb": 2, "fruit": 1, "fat": 1},
    "lunch": {"protein": 2, "carb": 2, "veg": 2, "fat": 1, "dairy": 1},
    "dinner": {"protein": 2, "carb": 2, "veg": 2, "fat": 1, "dairy": 1},
    "snack": {"protein": 1, "fat": 1, "fruit": 1, "snack": 1, "dairy": 1, "carb": 1, "veg": 1},
}
ITEM_BOUNDS = {"breakfast": (2, 4), "lunch": (3, 5), "dinner": (3, 5), "snack": (1, 2)}
# (daily kcal tolerance, per-meal band, enforce meal structure), tried in order
LADDER = [(0.03, 0.15, True), (0.05, 0.25, True), (0.08, 0.40, True), (0.10, 0.50, False)]
# How many candidate foods per role each meal considers (a seeded sample keeps the
# program small and fast, and is also what makes "another plan" differ).
SAMPLE_SIZES = {
    "breakfast": {"protein": 4, "dairy": 3, "carb": 5, "fruit": 4, "fat": 3},
    "lunch": {"protein": 6, "carb": 4, "veg": 5, "dairy": 2, "fat": 2},
    "dinner": {"protein": 6, "carb": 4, "veg": 5, "dairy": 2, "fat": 2},
    "snack": {"protein": 3, "fruit": 3, "fat": 2, "snack": 4, "dairy": 2, "carb": 1, "veg": 1},
}
MIN_VEG_GRAMS = 120.0    # vegetables in each lunch and dinner
SOLVE_SECONDS = 1.0      # per attempt
BUDGET_SECONDS = 6.0     # whole request


BULKY_UNITS = {"cup", "glass", "katori", "plate", "bowl", "serving"}
# Close variants count as one food per day (no white and whole-wheat bread in the same plan).
FAMILIES = {
    "white bread": "bread", "whole wheat bread": "bread",
    "white rice (cooked)": "rice", "brown rice (cooked)": "rice",
    "milk (whole)": "milk", "milk (low-fat)": "milk", "milk (skim)": "milk",
    "soy milk (unsweetened)": "milk", "almond milk (unsweetened)": "milk",
    "chicken breast (cooked)": "chicken", "chicken thigh (cooked)": "chicken",
}


def min_steps(food: Food) -> int:
    """Smallest realistic portion, in steps.

    Bulky foods start at half a serving. Countable foods start at about 100 kcal, so a plan
    never contains one idli or a single slice of bread. Fats and egg whites are left alone.
    """
    if food.display == "unit" and food.unit not in BULKY_UNITS:
        if food.role == "fat" or food.unit == "white":
            return 1
        per_step = food.kcal * food.step_g / 100
        return max(1, min(food.max_steps, math.ceil(100 / max(per_step, 1) - 1e-9)))
    if food.unit_g < 100 and food.unit not in BULKY_UNITS:
        return 1
    share = 0.5 if food.unit in BULKY_UNITS else 0.4
    return max(1, min(food.max_steps, math.ceil(share * food.unit_g / food.step_g - 1e-9)))


def _solve(slots, pools: list[list[Food]], targets: dict, noise: dict,
           tol: float, band: float, structure: bool,
           seconds: float = SOLVE_SECONDS) -> Optional[list[list[tuple[Food, int]]]]:
    variables = [(s, f) for s, pool in enumerate(pools) for f in pool]
    m = len(variables)
    if m == 0:
        return None
    nv = 2 * m + 9
    dp = lambda k: 2 * m + k            # overshoot of macro k
    dn = lambda k: 2 * m + 4 + k        # undershoot of macro k
    fs = 2 * m + 8                      # fibre shortfall

    steps = np.array([f.step_g for _, f in variables])[:, None] / 100.0
    per = np.array([[f.kcal, f.protein, f.carbs, f.fat, f.fiber] for _, f in variables]) * steps
    goal = np.array([targets["calories"], targets["protein_g"], targets["carbs_g"], targets["fat_g"]], float)
    goal = np.maximum(goal, 1.0)

    cost = np.zeros(nv)
    cost[m:2 * m] = [0.01 + noise[(s, f.id)] for s, f in variables]
    weights = np.array([4.0, 3.0, 1.0, 1.0]) / goal
    cost[2 * m:2 * m + 4] = weights
    cost[2 * m + 4:2 * m + 8] = weights
    cost[fs] = 2.0 / max(targets.get("fiber_g", 25), 1)

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add(row: np.ndarray, lb: float, ub: float) -> None:
        rows.append(row)
        lo.append(lb)
        hi.append(ub)

    for k in range(4):
        row = np.zeros(nv)
        row[:m] = per[:, k]
        row[dp(k)], row[dn(k)] = -1.0, 1.0
        add(row, goal[k], goal[k])
    row = np.zeros(nv)
    row[:m] = per[:, 0]
    add(row, (1 - tol) * goal[0], (1 + tol) * goal[0])
    row = np.zeros(nv)
    row[:m] = per[:, 4]
    row[fs] = 1.0
    add(row, float(targets.get("fiber_g", 25)), np.inf)

    for i, (_, f) in enumerate(variables):
        row = np.zeros(nv)
        row[i], row[m + i] = 1.0, -float(f.max_steps)
        add(row, -np.inf, 0.0)
        row = np.zeros(nv)
        row[i], row[m + i] = 1.0, -float(min_steps(f))
        add(row, 0.0, np.inf)

    for s, (slot, _, ratio) in enumerate(slots):
        members = [i for i, (si, _) in enumerate(variables) if si == s]
        row = np.zeros(nv)
        row[members] = per[members, 0]
        add(row, (1 - band) * ratio * goal[0], (1 + band) * ratio * goal[0])
        row = np.zeros(nv)
        row[[m + i for i in members]] = 1.0
        add(row, *ITEM_BOUNDS[slot])
        if structure:
            if slot in ("lunch", "dinner"):
                row = np.zeros(nv)
                for i in members:
                    if variables[i][1].role == "veg":
                        row[i] = variables[i][1].step_g
                add(row, MIN_VEG_GRAMS, np.inf)
            for roles, count in REQUIRED_ROLES[slot]:
                row = np.zeros(nv)
                row[[m + i for i in members if variables[i][1].role in roles]] = 1.0
                add(row, float(count), np.inf)
        for role, cap in ROLE_CAPS[slot].items():
            idx = [m + i for i in members if variables[i][1].role == role]
            if len(idx) > cap:
                row = np.zeros(nv)
                row[idx] = 1.0
                add(row, -np.inf, float(cap))

    by_food: dict[object, list[int]] = {}
    for i, (_, f) in enumerate(variables):
        by_food.setdefault(FAMILIES.get(f.name, f.id), []).append(m + i)
    for idx in by_food.values():
        if len(idx) > 1:
            row = np.zeros(nv)
            row[idx] = 1.0
            add(row, -np.inf, 1.0)

    lower = np.zeros(nv)
    upper = np.full(nv, np.inf)
    upper[:m] = [f.max_steps for _, f in variables]
    upper[m:2 * m] = 1.0
    integrality = np.zeros(nv)
    integrality[:2 * m] = 1

    res = milp(cost, constraints=LinearConstraint(np.vstack(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lower, upper), options={"time_limit": seconds, "mip_rel_gap": 0.10})
    if res.x is None:
        return None
    meals: list[list[tuple[Food, int]]] = [[] for _ in slots]
    for i, (s, f) in enumerate(variables):
        n = int(round(res.x[i]))
        if n > 0:
            meals[s].append((f, n))
    return meals


def _sample_pool(pool: list[Food], slot: str, rng: random.Random) -> list[Food]:
    sizes = SAMPLE_SIZES[slot]
    picked: list[Food] = []
    for role, k in sizes.items():
        group = [f for f in pool if f.role == role]
        rng.shuffle(group)
        picked.extend(group[:k])
    return picked


def _penalty(plan: dict) -> float:
    """How far a plan is from its targets, in percentage points (calories count double)."""
    return abs(plan["kcal_error_pct"]) + 0.5 * abs(plan["protein_error_pct"])


GOOD_ENOUGH = 4.0


def plan_day(db: FoodDB, targets: dict, diet: Optional[str] = None,
             allergens: Iterable[str] = (), avoid: Iterable[int] = (),
             meals: int = 4, seed: int = 0) -> Optional[dict]:
    """Return a day of meals close to the targets, or None if nothing satisfies the restrictions.

    The first pass uses a seeded sample of foods (fast, varied). If that misses the targets by
    more than a few percent, a second pass uses every allowed food and the better plan wins.
    """
    slots = SLOT_PLANS.get(meals, SLOT_PLANS[4])
    full = [db.allowed(diet, allergens, avoid, slot=slot) for slot, _, _ in slots]
    rng = random.Random(seed)
    deadline = time.monotonic() + BUDGET_SECONDS
    best: Optional[dict] = None
    for sampled in (True, False):
        if best is not None and _penalty(best) <= GOOD_ENOUGH:
            break
        pools = [_sample_pool(p, slot, rng) for p, (slot, _, _) in zip(full, slots)] if sampled else full
        noise = {(s, f.id): rng.uniform(0.0, 0.02) for s, pool in enumerate(pools) for f in pool}
        for tol, band, structure in LADDER:
            left = deadline - time.monotonic()
            if left <= 0.2:
                return best
            solution = _solve(slots, pools, targets, noise, tol, band, structure, min(SOLVE_SECONDS, left))
            if solution is not None and any(solution):
                plan = _assemble(slots, solution, targets)
                if best is None or _penalty(plan) < _penalty(best):
                    best = plan
                break
    return best


def _assemble(slots, solution, targets: dict) -> dict:
    meals_out = []
    day = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
    for (slot, title, _), items in zip(slots, solution):
        items = sorted(items, key=lambda p: (ROLE_ORDER.index(p[0].role), p[0].name))
        rows, tot = [], {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
        for food, n in items:
            grams = n * food.step_g
            rows.append(food.summary(grams))
            for k, v in food.nutrients(grams).items():
                tot[k] += v
        for k in day:
            day[k] += tot[k]
        meals_out.append({"name": title, "slot": slot, "items": rows,
                          "totals": {k: round(v, 1) if k != "kcal" else round(v) for k, v in tot.items()}})
    goal_kcal = max(targets["calories"], 1)
    goal_protein = max(targets.get("protein_g", 1), 1)
    return {
        "meals": meals_out,
        "totals": {k: round(v, 1) if k != "kcal" else round(v) for k, v in day.items()},
        "targets": {k: targets[k] for k in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g") if k in targets},
        "kcal_error_pct": round((day["kcal"] - goal_kcal) / goal_kcal * 100, 1),
        "protein_error_pct": round((day["protein"] - goal_protein) / goal_protein * 100, 1),
    }
