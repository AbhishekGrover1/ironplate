from app.nlu import CONFIDENCE_MIN, classifier_text
from app.training import build_dataset

CASES = [
    ("hello there coach", "greet"), ("good morning", "greet"), ("thank you so much, that was really helpful", "thanks"),
    ("bye for now", "bye"), ("what do you offer", "help"), ("who developed you", "about"),
    ("i'm vegetarian and allergic to nuts", "set_profile"), ("change my goal to building muscle", "set_profile"),
    ("how many calories should i be eating", "calc_targets"), ("what is my bmi", "calc_targets"),
    ("make me a meal plan", "meal_plan"), ("plan my food for today", "meal_plan"), ("another plan please", "meal_plan"),
    ("create a workout plan for me", "workout_plan"), ("i need a 4 day gym split", "workout_plan"),
    ("i had 2 eggs and 2 slices of toast for breakfast", "log_food"), ("ate dal rice and curd for lunch", "log_food"),
    ("log a banana", "log_food"), ("just had a protein shake", "log_food"),
    ("how many calories are in a banana", "food_info"), ("protein in paneer", "food_info"), ("nutrition of chicken breast", "food_info"),
    ("give me some high protein snack ideas", "food_suggest"), ("what can i eat instead of white rice", "food_suggest"),
    ("i weighed 77.4 kg this morning", "log_weight"), ("scale says 165 lbs today", "log_weight"),
    ("how is my progress", "progress"), ("am i on track for my goal", "progress"),
    ("what have i eaten so far today", "today_summary"), ("how many calories do i have left", "today_summary"),
    ("how do i do a proper squat", "exercise_info"), ("best chest exercises", "exercise_info"),
    ("delete all my data", "reset"), ("what is the capital of japan", "out_of_scope"), ("tell me a funny joke", "out_of_scope"),
    ("wat shud i eat today", "meal_plan"), ("calaries in banana", "food_info"), ("gym plan 4 days", "workout_plan"),
]


def test_fresh_phrases_are_classified(assistant):
    wrong = []
    for text, want in CASES:
        got, conf, _ = assistant.model.predict(text)
        if got != want or conf < CONFIDENCE_MIN:
            wrong.append((text, want, got, round(conf, 2)))
    assert len(wrong) <= 2, wrong


def test_gibberish_falls_below_the_confidence_threshold(assistant):
    _, conf, _ = assistant.model.predict("asdfgh qwerty zxcv")
    assert conf < 0.6


def test_classifier_text_masks_numbers():
    assert classifier_text("I weigh 77.4 kg") == "i weigh num kg"
    assert classifier_text("my height is 5'10") == "my height is numft"


def test_training_data_covers_every_intent_with_groups():
    texts, labels, groups = build_dataset(seed=1, per_template=4)
    assert len(texts) == len(labels) == len(groups) > 1500
    assert len(set(labels)) == 18
    # variants of one phrase must share a group so cross-validation cannot leak
    assert len(set(groups)) < len(texts)
