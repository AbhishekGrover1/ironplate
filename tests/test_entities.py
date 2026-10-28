import pytest

from app.entities import extract


def test_full_profile_sentence():
    e = extract("I'm 26, male, 178 cm, 78 kg, want to lose fat, train 4 days a week")
    assert (e["age"], e["sex"], e["height_cm"], e["weight_kg"]) == (26, "male", 178.0, 78.0)
    assert e["goal"] == "lose" and e["days"] == 4


def test_imperial_units_are_converted():
    e = extract("i am a 24 year old female, 5'4, 130 lbs and i want to build muscle")
    assert e["age"] == 24 and e["sex"] == "female" and e["goal"] == "gain"
    assert round(e["height_cm"]) == 163 and round(e["weight_kg"]) == 59 and e["units"] == "imperial"


@pytest.mark.parametrize("text,key,value", [
    ("age 27 male 80kg 175cm", "age", 27), ("5 ft 10 inches 150 lbs 31 male", "age", 31),
    ("1.75 m and 68 kg", "height_cm", 175.0), ("weighed in at 80.2 kg", "weight_kg", 80.2),
    ("i am vegan", "diet", "vegan"), ("non veg", "diet", "nonveg"), ("i'm vegetarian", "diet", "vegetarian"),
    ("i have a desk job", "activity", "sedentary"), ("4 meals a day", "meals", 4), ("aggressive fat loss", "pace", "fast"),
    ("a 3-5 days a week routine", "days", 4), ("i'm a beginner with dumbbells at home", "experience", "beginner"),
])
def test_single_slots(text, key, value):
    assert extract(text)[key] == value


def test_target_weight_is_not_current_weight():
    e = extract("I weigh 80 kg and want to get to 72 kg")
    assert e["weight_kg"] == 80.0 and e["goal_weight_kg"] == 72.0


def test_relative_goal_and_timeframe():
    e = extract("i want to lose 10 kg in 2 weeks")
    assert e["goal_delta_kg"] == -10.0 and e["timeframe_weeks"] == 2 and e["goal"] == "lose"
    assert "weight_kg" not in e


def test_allergens():
    assert extract("i'm allergic to nuts")["allergens"] == ["nuts", "peanut"]
    assert extract("i am lactose intolerant and gluten free")["allergens"] == ["dairy", "gluten"]
    assert extract("clear my restrictions")["allergens"] == []


def test_conflicting_goals_become_maintenance():
    assert extract("i want to lose fat and build muscle")["goal"] == "maintain"


def test_bare_numbers_answer_the_question_asked():
    assert extract("178", expecting=["height_cm", "weight_kg"]) == {"height_cm": 178.0}
    both = extract("178 78", expecting=["height_cm", "weight_kg"])
    assert both["height_cm"] == 178.0 and both["weight_kg"] == 78.0
    assert extract("170 lbs", expecting=["weight_kg"])["weight_kg"] == pytest.approx(77.1, abs=0.1)


def test_out_of_range_values_are_flagged_not_used():
    e = extract("i weigh 600 kg and i'm 300 cm tall")
    assert "weight_kg" not in e and "height_cm" not in e and set(e["invalid"]) == {"height_cm"} | set()


def test_ordinary_sentences_extract_nothing():
    assert extract("I had 2 eggs for breakfast") == {}
    assert "age" not in extract("i have trained for 2 years")
