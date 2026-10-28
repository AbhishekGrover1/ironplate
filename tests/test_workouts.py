import pytest

from app.workouts import EQUIPMENT_SETS, ExerciseDB, build_plan

DB = ExerciseDB.load()


@pytest.mark.parametrize("days", [1, 2, 3, 4, 5, 6])
def test_plan_has_one_entry_per_day(days):
    plan = build_plan(DB, days=days, equipment="gym", level="intermediate", seed=1)
    assert len(plan["days"]) == days and all(d["exercises"] for d in plan["days"])


@pytest.mark.parametrize("equipment", ["gym", "home", "bodyweight"])
def test_only_available_equipment_is_used(equipment):
    allowed = EQUIPMENT_SETS[equipment]
    by_name = {e.label: e for e in DB.items}
    plan = build_plan(DB, days=4, equipment=equipment, level="beginner", seed=2)
    for day in plan["days"]:
        for ex in day["exercises"]:
            assert by_name[ex["name"]].equipment in allowed


def test_no_exercise_repeats_within_a_day():
    plan = build_plan(DB, days=5, equipment="gym", level="advanced", seed=3)
    for day in plan["days"]:
        names = [e["name"] for e in day["exercises"]]
        assert len(names) == len(set(names))


def test_beginners_avoid_the_hardest_lifts():
    by_name = {e.label: e for e in DB.items}
    for seed in range(5):
        plan = build_plan(DB, days=3, equipment="gym", level="beginner", seed=seed)
        assert all(by_name[e["name"]].level <= 2 for d in plan["days"] for e in d["exercises"])


def test_search_and_muscles():
    assert DB.search("deadlift")[0].name == "conventional deadlift"
    assert DB.search("romanian dedlift")[0].name == "romanian deadlift"
    assert DB.muscles_in("best exercises for chest and legs") == ["chest", "quads", "hamstrings", "glutes", "calves"]
    assert all(e.primary == "chest" for e in DB.for_muscle("chest", "home"))
