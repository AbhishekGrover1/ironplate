import random
from datetime import date, timedelta

from app.trend import analyze, clean_series, ewma


def history(days=35, true_tdee=2500, eaten=2000, noise=0.3, start_kg=80.0, seed=3):
    rng = random.Random(seed)
    start = date(2026, 9, 1)
    kg, weights, intake = start_kg, [], {}
    for i in range(days):
        d = start + timedelta(days=i)
        kg -= (true_tdee - eaten) / 7700
        weights.append((d.isoformat(), kg + rng.gauss(0, noise)))
        if i % 7 != 3:
            intake[d] = eaten + rng.gauss(0, 100)
    return clean_series(weights), intake


def test_needs_a_few_points():
    assert analyze(clean_series([("2026-10-01", 80), ("2026-10-02", 79.9)]), {})["status"] == "need_more"


def test_rate_is_recovered_despite_noise():
    weights, intake = history()
    result = analyze(weights, intake, goal="lose")
    assert result["status"] == "ok"
    assert abs(result["weekly_rate_kg"] - (-500 / 7700 * 7)) < 0.15
    assert "on_track" in result["flags"]


def test_adaptive_maintenance_is_close_to_truth():
    weights, intake = history()
    adaptive = analyze(weights, intake, goal="lose", calorie_offset=-400)["adaptive"]
    assert adaptive is not None and abs(adaptive["tdee_est"] - 2500) < 250
    assert abs(adaptive["suggested_calories"] - (adaptive["tdee_est"] - 400)) <= 10


def test_flags_plateau_and_too_fast():
    flat, _ = history(eaten=2500, noise=0.1)
    assert "plateau" in analyze(flat, {}, goal="lose")["flags"]
    fast, _ = history(eaten=1000, noise=0.1)
    assert "too_fast" in analyze(fast, {}, goal="lose")["flags"]


def test_eta_to_goal():
    weights, _ = history(noise=0.1)
    result = analyze(weights, {}, goal="lose", goal_weight=70)
    assert result["eta_weeks"] and 5 <= result["eta_weeks"] <= 40


def test_series_cleaning():
    out = clean_series([("2026-10-02", 80), ("2026-10-01", 81), ("2026-10-02", 79.5), ("bad", 70), ("2026-10-03", 5000)])
    assert out == [(date(2026, 10, 1), 81.0), (date(2026, 10, 2), 79.5)]


def test_ewma_smooths():
    pts = clean_series([("2026-10-01", 80), ("2026-10-02", 82), ("2026-10-03", 80)])
    trend = ewma(pts)
    assert trend[0] == 80 and 80 < trend[1] < 82
