import pytest

from app.foods import FoodDB


@pytest.fixture(scope="module")
def db():
    return FoodDB.load()


def test_table_is_consistent(db):
    assert len(db.foods) > 100
    for f in db.foods:
        assert f.unit_g > 0 and f.step_g > 0 and f.max_steps >= 1
        assert abs(f.kcal - (4 * f.protein + 4 * f.carbs + 9 * f.fat)) <= 0.12 * f.kcal + 12, f.name


@pytest.mark.parametrize("query,expected", [
    ("banan", "banana"), ("eggs", "egg (boiled)"), ("protien shake", "whey protein"), ("milk", "milk (whole)"),
    ("chapati", "roti"), ("biriyani", "chicken biryani"), ("tomatoes", "tomato"), ("dahi", "curd (plain)"),
])
def test_search_handles_typos_plurals_and_aliases(db, query, expected):
    food, score = db.match(query)
    assert food.name == expected and score >= 0.5


@pytest.mark.parametrize("query", ["hello", "quantum", "xyz"])
def test_search_rejects_unrelated_words(db, query):
    assert db.match(query) is None


def names(parsed):
    return [i.food.name for i in parsed.items]


def test_parse_meal_with_commas_units_and_meal_name(db):
    r = db.parse_meal("I had 2 eggs, 2 slices of toast and a banana for breakfast")
    assert r.meal == "breakfast"
    assert names(r) == ["egg (boiled)", "white bread", "banana"]
    assert round(r.items[0].grams) == 100 and round(r.items[1].grams) == 56


def test_parse_meal_without_separators(db):
    assert names(db.parse_meal("2 rotis dal and rice")) == ["roti", "dal (cooked)", "white rice (cooked)"]


def test_parse_meal_quantities(db):
    grams = lambda text: round(db.parse_meal(text).items[0].grams)
    assert grams("200g chicken breast") == 200
    assert grams("1 cup of rice") == 158
    assert grams("1 and a half cups of rice") == 237
    assert grams("half a banana") == 59
    assert grams("a glass of milk") == 244
    assert grams("2 tbsp peanut butter") == 32


def test_protected_phrase_is_not_split(db):
    assert names(db.parse_meal("had tea with milk and 2 biscuits")) == ["chai (milk tea)", "biscuit (tea)"]


def test_unmatched_is_reported(db):
    r = db.parse_meal("had asdfgh and a banana")
    assert names(r) == ["banana"] and r.unmatched == ["asdfgh"]


def test_meal_defaults_from_hour(db):
    assert db.parse_meal("an apple", hour=8).meal == "breakfast"
    assert db.parse_meal("an apple", hour=20).meal == "dinner"


def test_allowed_respects_diet_and_allergens(db):
    vegan = db.allowed("vegan")
    assert vegan and all(f.diet == "vegan" for f in vegan)
    no_dairy = db.allowed("vegetarian", ["dairy"])
    assert all("dairy" not in f.allergens for f in no_dairy)


def test_swaps_stay_in_role(db):
    rice = db.match("rice")[0]
    near = db.similar(rice, db.allowed(), 5)
    assert near and all(f.role == rice.role and f.id != rice.id for f, _ in near)
