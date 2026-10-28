from app.nutrition import compute_targets, missing_fields

BASE = dict(sex="male", age=30, height_cm=180, weight_kg=80)


def test_bmr_matches_mifflin_st_jeor():
    # 10*80 + 6.25*180 - 5*30 + 5 = 1780
    assert compute_targets(BASE).bmr == 1780
    assert compute_targets({**BASE, "sex": "female"}).bmr == 1614


def test_macros_add_up_to_calories():
    for goal in ("lose", "maintain", "gain"):
        t = compute_targets({**BASE, "goal": goal})
        assert abs(4 * t.protein_g + 4 * t.carbs_g + 9 * t.fat_g - t.calories) < 40


def test_goal_direction():
    lose = compute_targets({**BASE, "goal": "lose"})
    gain = compute_targets({**BASE, "goal": "gain"})
    assert lose.calories < lose.tdee < gain.calories
    assert lose.weekly_change_kg < 0 < gain.weekly_change_kg


def test_loss_is_capped_at_one_percent_of_body_weight_per_week():
    t = compute_targets({**BASE, "goal": "lose", "pace": "fast"})
    assert abs(t.weekly_change_kg) <= 0.01 * 80 + 1e-6


def test_calorie_floor_and_underweight_guard():
    small = compute_targets(dict(sex="female", age=60, height_cm=150, weight_kg=45, goal="lose", activity="sedentary"))
    assert small.calories >= min(1200, small.tdee)
    under = compute_targets(dict(sex="female", age=25, height_cm=170, weight_kg=48, goal="lose"))
    assert under.goal == "maintain" and any("underweight" in n for n in under.notes)


def test_activity_follows_training_days():
    assert compute_targets({**BASE, "days": 1}).activity == "light"
    assert compute_targets({**BASE, "days": 4}).activity == "moderate"
    assert compute_targets({**BASE, "days": 6}).activity == "active"


def test_missing_fields():
    assert missing_fields({"age": 20}) == ["sex", "height_cm", "weight_kg"]
    assert missing_fields(BASE) == []
