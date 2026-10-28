import pytest


def block_types(out):
    return [b["type"] for b in out["blocks"]]


def test_details_in_one_message_give_targets(client):
    out = client.say("I'm 26, male, 178 cm, 78 kg, want to lose fat, train 4 days a week")
    assert out["intent"] == "set_profile" and "targets" in block_types(out)
    assert 2200 <= out["targets"]["calories"] <= 2450
    assert client.profile["goal"] == "lose" and client.profile["days"] == 4


def test_missing_details_are_asked_for_and_the_request_resumes(client):
    out = client.say("make me a meal plan")
    assert client.pending["type"] == "need_profile" and "age, sex, height and weight" in out["reply"]
    out = client.say("26, male, 178 cm")
    assert "still need your weight" in out["reply"] and client.pending["missing"] == ["weight_kg"]
    out = client.say("78")
    assert "meal_plan" in block_types(out) and client.pending is None


def test_imperial_input_is_understood(client):
    client.say("I'm 29, female, 5'4, 130 lbs, want to lose weight")
    assert round(client.profile["height_cm"]) == 163 and round(client.profile["weight_kg"]) == 59
    assert client.profile["units"] == "imperial"


def test_logging_food_tracks_what_is_left(client):
    client.say("I'm 26, male, 178 cm, 78 kg, want to lose fat")
    out = client.say("I had 2 eggs and a banana for breakfast")
    assert out["intent"] == "log_food" and "Today so far" in out["reply"]
    assert out["actions"][0]["type"] == "log_food" and len(client.log[client.today]) == 2
    summary = client.say("what's left today")
    assert "today" in block_types(summary) and "kcal so far" in summary["reply"]


def test_single_food_name_is_a_lookup_not_a_log(client):
    out = client.say("paneer")
    assert out["intent"] == "food_info" and not out["actions"] and "food_info" in block_types(out)
    assert not client.log


def test_restrictions_are_saved_and_honoured(client, assistant):
    client.say("I'm 29, female, 163 cm, 59 kg, vegetarian, allergic to nuts")
    assert client.profile["diet"] == "vegetarian" and client.profile["allergens"] == ["nuts", "peanut"]
    out = client.say("plan my meals")
    plan = next(b for b in out["blocks"] if b["type"] == "meal_plan")
    for meal in plan["meals"]:
        for item in meal["items"]:
            food = assistant.foods.by_id[item["id"]]
            assert food.diet in ("vegan", "vegetarian") and not (food.allergens & {"nuts", "peanut"})


def test_dislikes_are_remembered_not_logged(client):
    out = client.say("i hate mushrooms and no paneer")
    assert out["intent"] == "set_profile" and len(client.profile["avoid"]) == 2 and not out["actions"]


def test_another_plan_differs(client):
    client.say("I'm 26, male, 178 cm, 78 kg, want to lose fat")
    first = client.say("meal plan")["blocks"][0]
    second = client.say("another meal plan")["blocks"][0]
    ids = lambda p: {i["id"] for m in p["meals"] for i in m["items"]}
    assert ids(first) != ids(second)


@pytest.mark.parametrize("text,kind", [
    ("i want to kill myself", "crisis"), ("chest pain when i run", "medical"), ("i've been bingeing again", "eating"),
    ("i am pregnant, calculate my calories", "pregnancy"), ("i feel so sad today", "mood"),
])
def test_safety_messages_never_get_plans(client, text, kind):
    out = client.say(text)
    assert out["intent"] == f"safety_{kind}" and not out["blocks"] and out["targets"] is None


def test_minors_do_not_get_calorie_targets(client):
    out = client.say("I'm 16, male, 170 cm, 60 kg, plan my meals")
    assert out["targets"] is None and "18 and over" in out["reply"] and "meal_plan" not in block_types(out)


def test_reset_needs_confirmation(client):
    client.say("I'm 26, male, 178 cm, 78 kg")
    client.say("reset my data")
    assert client.pending == {"type": "confirm_reset"}
    client.say("no")
    assert client.profile and client.pending is None
    client.say("reset my data")
    out = client.say("yes")
    assert out["actions"] == [{"type": "reset_all"}] and client.profile == {}


def test_faq_answers_questions_but_not_profile_statements(client):
    assert client.say("should i take creatine")["intent"] == "faq"
    assert client.say("I'm 26, male, 178 cm, 78 kg, want to lose fat, train 4 days a week")["intent"] == "set_profile"


def test_unrealistic_goals_are_challenged(client):
    out = client.say("i want to lose 10 kg in 2 weeks")
    assert "faster than" in out["reply"]


def test_goal_weight_in_the_underweight_range_is_refused(client):
    client.say("I'm 29, female, 163 cm, 59 kg")
    client.say("my target weight is 45 kg")
    assert "goal_weight_kg" not in client.profile


def test_progress_needs_data_then_reports_a_trend(client):
    client.say("I'm 26, male, 178 cm, 80 kg, want to lose fat")
    assert "at least three weigh-ins" in client.say("how am I doing")["reply"]
    for i, kg in enumerate([80.0, 79.8, 79.5, 79.3, 79.0, 78.8, 78.5, 78.3]):
        client.weights[client.days_ago(14 - 2 * i)] = kg
    out = client.say("how am I doing")
    assert "trend" in block_types(out) and "kg a week" in out["reply"]


def test_weight_logging_updates_profile_and_history(client):
    client.say("I'm 26, male, 178 cm, 78 kg")
    out = client.say("i weighed 77.4 kg this morning")
    assert client.profile["weight_kg"] == 77.4 and client.weights[client.today] == 77.4 and "Logged" in out["reply"]


def test_exercise_questions(client):
    assert "exercise" in block_types(client.say("how to do a deadlift"))
    assert "exercise_list" in block_types(client.say("chest exercises"))
    assert "exercise_list" in block_types(client.say("leg day"))


def test_food_suggestions_and_swaps(client):
    out = client.say("high protein vegetarian snacks")
    assert "foods_list" in block_types(out)
    swap = client.say("what can i eat instead of rice")
    assert "foods_list" in block_types(swap) and "swaps" in swap["reply"].lower()


def test_unknown_text_gets_a_helpful_fallback(client):
    out = client.say("asdfgh qwerty zxcv")
    assert "I can help with food and training" in out["reply"]


def test_about_names_the_author(client):
    assert "Abhishek Grover" in client.say("who built you")["reply"]
