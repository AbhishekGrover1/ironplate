import pytest

from app.foods import DIET_LEVEL, FoodDB
from app.planner import plan_day

DB = FoodDB.load()


def macros(kcal):
    p = round(0.25 * kcal / 4)
    f = round(0.28 * kcal / 9)
    return dict(calories=kcal, protein_g=p, carbs_g=round((kcal - 4 * p - 9 * f) / 4), fat_g=f,
                fiber_g=max(20, round(14 * kcal / 1000)))


@pytest.mark.parametrize("diet,kcal,meals", [("nonveg", 2200, 4), ("vegetarian", 1800, 3), ("vegan", 2600, 5)])
def test_plan_hits_targets(diet, kcal, meals):
    target = macros(kcal)
    plan = plan_day(DB, target, diet=diet, meals=meals, seed=3)
    assert plan is not None and len(plan["meals"]) == meals
    assert abs(plan["kcal_error_pct"]) <= 3
    assert abs(plan["protein_error_pct"]) <= 10


def test_plan_respects_restrictions():
    banned = ["dairy", "gluten"]
    plan = plan_day(DB, macros(2000), diet="vegetarian", allergens=banned, meals=4, seed=1)
    assert plan is not None
    for meal in plan["meals"]:
        for item in meal["items"]:
            food = DB.by_id[item["id"]]
            assert DIET_LEVEL[food.diet] <= DIET_LEVEL["vegetarian"]
            assert not (food.allergens & set(banned)), food.name


def test_plan_respects_disliked_foods():
    paneer = DB.match("paneer")[0].id
    plan = plan_day(DB, macros(2000), diet="vegetarian", avoid=[paneer], meals=4, seed=2)
    assert all(item["id"] != paneer for meal in plan["meals"] for item in meal["items"])


def test_meals_have_structure_and_no_repeats():
    plan = plan_day(DB, macros(2200), diet="nonveg", meals=4, seed=4)
    seen = []
    for meal in plan["meals"]:
        roles = {DB.by_id[i["id"]].role for i in meal["items"]}
        if meal["slot"] in ("lunch", "dinner"):
            assert {"protein", "carb", "veg"} <= roles
        seen += [i["id"] for i in meal["items"]]
    assert len(seen) == len(set(seen))


def test_seed_changes_the_plan():
    a = plan_day(DB, macros(2200), diet="nonveg", meals=4, seed=1)
    b = plan_day(DB, macros(2200), diet="nonveg", meals=4, seed=2)
    ids = lambda p: {i["id"] for m in p["meals"] for i in m["items"]}
    assert ids(a) != ids(b)


def test_impossible_restrictions_return_none():
    everything = ["dairy", "gluten", "nuts", "peanut", "soy", "egg", "fish", "shellfish", "sesame"]
    avoid = [f.id for f in DB.foods]
    assert plan_day(DB, macros(2000), diet="vegan", allergens=everything, avoid=avoid, meals=3, seed=0) is None
