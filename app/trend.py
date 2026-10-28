"""Weight trend analysis and adaptive maintenance-calorie estimate.

The trend line is a time-aware exponential moving average. The weekly rate is a
Theil-Sen slope (robust to water-weight spikes). Maintenance calories are
estimated from energy balance: average logged intake minus the energy implied
by the change in weight.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import numpy as np
from scipy.stats import theilslopes

from .nutrition import KCAL_PER_KG

WINDOW_DAYS = 42
MIN_POINTS = 3


def parse_day(value: object) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def clean_series(raw: Iterable[tuple[object, object]]) -> list[tuple[date, float]]:
    """Validate, de-duplicate by day (last entry wins) and sort."""
    by_day: dict[date, float] = {}
    for d, kg in raw:
        day = parse_day(d)
        try:
            val = float(kg)
        except (TypeError, ValueError):
            continue
        if day is None or not 25.0 <= val <= 400.0:
            continue
        by_day[day] = val
    return sorted(by_day.items())


def ewma(points: list[tuple[date, float]], tau_days: float = 7.0) -> list[float]:
    out: list[float] = []
    for i, (d, y) in enumerate(points):
        if i == 0:
            out.append(y)
            continue
        dt = max((d - points[i - 1][0]).days, 1)
        alpha = 1 - math.exp(-dt / tau_days)
        out.append(out[-1] + alpha * (y - out[-1]))
    return out


def analyze(weights: list[tuple[date, float]], intake: dict[date, float],
            goal: Optional[str] = None, goal_weight: Optional[float] = None,
            calorie_offset: Optional[float] = None) -> dict:
    """Summarise progress. `calorie_offset` is target calories minus formula maintenance."""
    if len(weights) < MIN_POINTS:
        return {"status": "need_more", "n": len(weights)}
    last_day = weights[-1][0]
    window = [(d, w) for d, w in weights if (last_day - d).days <= WINDOW_DAYS]
    span = (window[-1][0] - window[0][0]).days
    if len(window) < MIN_POINTS or span < 5:
        return {"status": "need_more", "n": len(weights)}

    trend_all = ewma(weights)
    xs = np.array([(d - window[0][0]).days for d, _ in window], dtype=float)
    ys = np.array([w for _, w in window], dtype=float)
    slope, _, low, high = theilslopes(ys, xs, alpha=0.9)
    trend_now = trend_all[-1]
    weekly = float(slope * 7)

    result: dict = {
        "status": "ok", "n": len(weights), "span_days": span,
        "points": [{"date": d.isoformat(), "kg": round(w, 1), "trend": round(t, 2)}
                   for (d, w), t in list(zip(weights, trend_all))[-60:]],
        "trend_kg": round(trend_now, 1), "weekly_rate_kg": round(weekly, 2),
        "rate_low": round(float(low * 7), 2), "rate_high": round(float(high * 7), 2),
        "pct_per_week": round(weekly / trend_now * 100, 2), "flags": [],
        "goal_weight_kg": goal_weight, "eta_weeks": None, "adaptive": None,
    }

    flags: list[str] = result["flags"]
    pct = weekly / trend_now * 100
    if goal == "lose":
        if pct < -1.0:
            flags.append("too_fast")
        elif weekly > 0.1:
            flags.append("wrong_direction")
        elif abs(weekly) < 0.1 and span >= 14:
            flags.append("plateau")
        else:
            flags.append("on_track")
    elif goal == "gain":
        if pct > 0.5:
            flags.append("too_fast")
        elif weekly < -0.1:
            flags.append("wrong_direction")
        elif abs(weekly) < 0.05 and span >= 14:
            flags.append("plateau")
        else:
            flags.append("on_track")
    elif abs(weekly) > 0.5:
        flags.append("drifting")
    else:
        flags.append("steady")

    if goal_weight and abs(weekly) >= 0.05:
        remaining = goal_weight - trend_now
        if remaining * weekly > 0:
            result["eta_weeks"] = min(int(math.ceil(remaining / weekly)), 104)

    # Adaptive maintenance estimate from the last four weeks.
    cutoff = last_day - timedelta(days=28)
    days_logged = {d: k for d, k in intake.items() if cutoff <= d <= last_day and k >= 800}
    win28 = [(d, w) for d, w in weights if d >= cutoff]
    span28 = (win28[-1][0] - win28[0][0]).days if len(win28) >= 2 else 0
    if len(win28) >= 5 and span28 >= 14 and len(days_logged) >= 10:
        x28 = np.array([(d - win28[0][0]).days for d, _ in win28], dtype=float)
        y28 = np.array([w for _, w in win28], dtype=float)
        s28, *_ = theilslopes(y28, x28, alpha=0.9)
        avg_intake = float(np.mean(list(days_logged.values())))
        est = avg_intake - float(s28) * KCAL_PER_KG
        coverage = len(days_logged) / (span28 + 1)
        if 1000 <= est <= 6000:
            result["adaptive"] = {
                "tdee_est": int(round(est / 10) * 10), "avg_intake": int(round(avg_intake)),
                "days_logged": len(days_logged),
                "confidence": "high" if coverage >= 0.8 and span28 >= 21 else "medium" if coverage >= 0.6 else "low",
                "suggested_calories": (int(round((est + calorie_offset) / 10) * 10)
                                       if calorie_offset is not None else None),
            }
    return result
