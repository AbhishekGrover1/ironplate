"""Rule-based slot extraction from free text.

Pulls age, sex, height, weight, goal, activity, diet, allergens and training
preferences out of messages such as "I'm 26, male, 5'10, 172 lb, want to lose fat".
Everything is returned in metric units. Out-of-range numbers are reported in
`invalid` instead of being used.
"""
from __future__ import annotations

import re
from typing import Optional

LB_TO_KG = 0.45359237
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
RANGES = {"age": (10, 100), "height_cm": (100, 250), "weight_kg": (25, 400)}

_WEIGHT = re.compile(
    r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilo|kilos|kilogram|kilograms|lb|lbs|pound|pounds)\b")
_WEIGHT_CUE = re.compile(r"\b(?:weigh|weighs|weighed|weighing|weight)\b[^0-9]{0,18}?(?<![\d.])(\d{2,3}(?:\.\d+)?)\b(?!\s*(?:cm|ft|feet|'|\"|years?|yo|y/o|yrs|%|kcal|calories|g\b|/))")
_DELTA = re.compile(
    r"\b(?:want to|wanna|plan to|aim to|goal is to|trying to|hope to|need to)\s+(lose|gain|drop|shed|cut|put on)\s+"
    r"(?:about |around |roughly )?(\d{1,3}(?:\.\d+)?)\s*(kg|kgs|kilos?|lb|lbs|pounds?)\b")
_TIMEFRAME = re.compile(r"\bin\s+(?:about\s+|around\s+|just\s+)?(\d{1,2})\s*(day|days|week|weeks|month|months)\b")
_TARGET_CUE = re.compile(r"(target|goal|reach|get to|get down to|down to|up to|want to be|aim|ideal|dream|to be at)[^0-9]{0,22}$")
_CM = re.compile(r"(?<![\d.])(\d{3}(?:\.\d+)?)\s*(?:cm|cms|centimeters?|centimetres?)\b")
_METRES = re.compile(r"(?<![\d.])([12]\.\d{1,2})\s*(?:m|meters?|metres?)\b")
_FEET_WORD = re.compile(r"\b([3-7])\s*(?:ft|feet|foot)\b\.?\s*(\d{1,2})?(?:\s*(?:in|inch|inches)\b)?")
_FEET_TICK = re.compile(r"\b([3-7])\s*'\s*(\d{1,2})?")
_HEIGHT_CUE = re.compile(r"\bheight\b[^0-9]{0,15}?(?<![\d.])(\d{3})\b(?!\s*(?:kg|lb|kcal|cal))")
_AGE_PATTERNS = [
    re.compile(r"(?<![\d.])(\d{1,2})\s*(?:years?|yrs?|yr)[\s-]*old\b"),
    re.compile(r"(?<![\d.])(\d{1,2})\s*(?:years?|yrs?)\b(?!\s*(?:of|experience|training|lifting|ago))"),
    re.compile(r"(?<![\d.])(\d{1,2})\s*(?:yo|y/o|yrs old|y\.o)\b"),
    re.compile(r"\bage(?:d)?\b[^0-9]{0,8}?(?<![\d.])(\d{1,2})\b(?!\s*(?:kg|lb|cm|ft|%))"),
    re.compile(r"\b(?:i am|i'm|im|iam)\s+(?:a\s+)?(\d{2})\b(?!\s*(?:kg|kgs|lb|lbs|cm|ft|feet|'|\"|inch|in\b|m\b|%|kilo|pound|day|days|time|times|min|mins))"),
    re.compile(r"(?<![\d.])(\d{2})\s*,?\s*(?:male|female|man|woman|guy|girl)\b"),
    re.compile(r"\b(?:male|female|man|woman|guy|girl)\s*,?\s*(\d{2})\b(?!\s*(?:kg|kgs|lb|lbs|cm|ft|feet|'|\"|inch|in\b|m\b|%|kilo|pound))"),
]

_GOAL_GAIN = re.compile(
    r"\b(gain(?:ing)?\s+(?:some\s+)?(?:muscle|weight|mass|size)|build(?:ing)?\s+(?:some\s+)?muscle|muscle\s+(?:gain|building)|"
    r"bulk(?:ing)?(?:\s+up)?|get\s+(?:bigger|jacked|stronger)|put\s+on\s+(?:weight|muscle|mass)|mass\s+gain|"
    r"weight\s+gain|hypertrophy|underweight)\b")
_GOAL_LOSE = re.compile(
    r"\b(los(?:e|ing)\s+(?:some\s+)?(?:weight|fat|belly|kilos|kgs|pounds)|(?:fat|weight)\s+loss|burn(?:ing)?\s+fat|"
    r"cutting|cut\s+phase|shed(?:ding)?|slim(?:\s+down)?|get\s+lean(?:er)?|lean\s+down|calorie\s+deficit|"
    r"drop(?:ping)?\s+(?:some\s+)?(?:weight|fat)|reduce\s+(?:my\s+)?(?:weight|fat|belly)|lose\s+it|"
    r"get\s+in\s+shape|trim\s+down|cut\s+fat)\b")
_GOAL_MAINTAIN = re.compile(
    r"\b(maintain(?:ing)?(?:\s+my)?(?:\s+weight)?|maintenance|recomp(?:osition)?|stay\s+the\s+same|keep\s+my\s+weight|"
    r"hold\s+my\s+weight|tone\s+up|toning|stay\s+fit|general\s+fitness)\b")

_ACTIVITY = [
    ("very_active", re.compile(r"\b(extremely active|extra active|athlete|two[- ]a[- ]days|twice a day|heavy (?:labou?r|manual)|construction worker)\b")),
    ("active", re.compile(r"\b(very active|highly active|6\s*(?:-|to)\s*7\s*(?:days|times)|train(?:ing)?\s+(?:every|each)\s+day|daily workouts?|physically demanding|active job)\b")),
    ("moderate", re.compile(r"\b(moderately active|moderate(?:ly)?\s+activ\w*|moderate exercise|3\s*(?:-|to)\s*5\s*(?:days|times)|3-5x)\b")),
    ("light", re.compile(r"\b(lightly active|light(?:ly)?\s+activ\w*|light exercise|walk(?:ing)? daily|1\s*(?:-|to)\s*3\s*(?:days|times)|once or twice a week|1-3x)\b")),
    ("sedentary", re.compile(r"\b(sedentary|desk job|office job|sit(?:ting)? all day|little to no exercise|no exercise|(?:do not|don't|dont) exercise|not (?:very )?active|barely exercise)\b")),
]
_DAYS = [
    re.compile(r"\b([1-7])\s*(?:-|to)?\s*(?:days?|x|times)\s*(?:a|per|/|each|every)?\s*(?:week|wk)\b"),
    re.compile(r"\b([1-7])[- ]day\s+(?:split|workout|plan|routine|program|programme|training|gym)\b"),
    re.compile(r"\b(?:train|training|workout|work out|lift|lifting|gym|exercise)\w*\s+(?:for\s+)?([1-7])\s+days?\b"),
    re.compile(r"\b(one|two|three|four|five|six|seven)\s+(?:days?|times)\s+(?:a|per)\s+week\b"),
    re.compile(r"\b([1-7])\s*(?:days?|times)\s+(?:a|per)\s+week\b"),
]
_DIETS = [
    ("nonveg", re.compile(r"\b(non[- ]?veg(?:etarian)?|omnivore|eat (?:everything|meat)|meat[- ]eater)\b")),
    ("vegan", re.compile(r"\b(vegan|plant[- ]based)\b")),
    ("eggetarian", re.compile(r"\b(eggetarian|ovo[- ]?vegetarian|veg(?:etarian)?\s+(?:but|with|plus)\s+eggs?|eat eggs? but no meat)\b")),
    ("pescatarian", re.compile(r"\b(pescatarian|pescetarian)\b")),
    ("vegetarian", re.compile(r"\b(vegetarian|pure veg|veggie|lacto[- ]?veg|veg)\b")),
]
_ALLERGENS = [
    ("dairy", re.compile(r"\b((?:lactose|dairy|milk)[- ]?(?:intoleran\w*|free|allerg\w*)|no dairy|avoid(?:ing)? dairy|without dairy|(?:can't|cant|cannot) (?:have|eat|digest|drink) (?:dairy|milk)|(?:don't|dont|do not) (?:eat|have|drink) (?:dairy|milk))\b")),
    ("gluten", re.compile(r"\b(gluten[- ]?(?:free|intoleran\w*|sensitiv\w*)|celiac|coeliac|no gluten|avoid(?:ing)? gluten|wheat allerg\w*)\b")),
    ("nuts", re.compile(r"\b((?:tree )?nuts? allerg\w*|allergic to (?:tree )?nuts?|no nuts?|nut[- ]free|avoid(?:ing)? nuts?|without nuts?)\b")),
    ("peanut", re.compile(r"\b(peanuts? allerg\w*|allergic to (?:peanuts?|nuts?)|no (?:peanuts?|nuts?)|peanut[- ]free|nuts? allerg\w*|nut[- ]free|avoid(?:ing)? (?:peanuts?|nuts?)|without (?:peanuts?|nuts?))\b")),
    ("soy", re.compile(r"\b(soy allerg\w*|allergic to soy|no soy|soy[- ]free|avoid(?:ing)? soy)\b")),
    ("egg", re.compile(r"\b(egg allerg\w*|allergic to eggs?|no eggs?|egg[- ]free|avoid(?:ing)? eggs?)\b")),
    ("fish", re.compile(r"\b(fish allerg\w*|allergic to fish|no fish|fish[- ]free)\b")),
    ("shellfish", re.compile(r"\b(shellfish allerg\w*|allergic to shellfish|no shellfish|prawn allerg\w*|shrimp allerg\w*)\b")),
    ("sesame", re.compile(r"\b(sesame allerg\w*|allergic to sesame|no sesame)\b")),
]
_CLEAR_ALLERGENS = re.compile(r"\b(no (?:more )?(?:allergies|restrictions)|remove (?:my )?(?:allergies|restrictions)|clear (?:my )?(?:allergies|restrictions))\b")
_MEALS = re.compile(r"\b([3-5])\s*(?:meals?|times a day)\b|\b(three|four|five)\s+meals?\b")
_PACE = re.compile(
    r"\b(?:(aggressive|fast|quick|rapid)(?:ly)?|(slow|gentle|steady|easy))\s*(?:fat |weight )?(?:loss|cut|pace|diet|deficit|bulk|gain|progress)\b"
    r"|\blos(?:e|ing)\s+(?:weight |fat )?(?:(fast|quickly|rapidly)|(slowly|gently|steadily))")
_EQUIPMENT = [
    ("bodyweight", re.compile(r"\b(no equipment|bodyweight|body weight|calisthenics|without equipment|no gym|no weights)\b")),
    ("home", re.compile(r"\b(at home|home workout|home gym|dumbbells?|resistance bands?|home)\b")),
    ("gym", re.compile(r"\b(gym|barbell|commercial gym|full gym|machines)\b")),
]
_EXPERIENCE = [
    ("beginner", re.compile(r"\b(beginner|just start(?:ed|ing)|new to (?:the )?(?:gym|lifting|working out|training|exercise)|never (?:lifted|trained|worked out)|novice)\b")),
    ("advanced", re.compile(r"\b(advanced|experienced|(?:[4-9]|\d{2})\+?\s*years?\s*(?:of\s*)?(?:lifting|training|gym|experience)|competitive|powerlifter|bodybuilder)\b")),
    ("intermediate", re.compile(r"\b(intermediate|[1-3]\s*years?\s*(?:of\s*)?(?:lifting|training|gym|experience))\b")),
]
_FOCUS = [
    ("strength", re.compile(r"\b(strength|powerlifting|get stronger|heavy)\b")),
    ("hypertrophy", re.compile(r"\b(hypertrophy|bodybuilding|muscle size|aesthetic|build muscle|muscle gain)\b")),
    ("general", re.compile(r"\b(general fitness|overall fitness|stay fit)\b")),
]
_SEX_MALE = re.compile(r"\b(male|man|guy|boy|gentleman)\b")
_SEX_FEMALE = re.compile(r"\b(female|woman|girl|lady)\b")
_SEX_OTHER = re.compile(r"\b(non[- ]?binary|prefer not to say|prefer not to answer)\b")


def clean(text: str) -> str:
    t = text.lower().replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"')
    t = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", t)
    return re.sub(r"\s+", " ", t).strip()


def _in_range(field: str, value: float) -> bool:
    lo, hi = RANGES[field]
    return lo <= value <= hi


def extract_weights(t: str, imperial_hint: bool = False) -> tuple[Optional[float], Optional[float], bool]:
    """Return (current_kg, target_kg, used_imperial) from explicit or cued weights."""
    found: list[tuple[int, float, bool]] = []
    spans = []
    for m in _WEIGHT.finditer(t):
        value = float(m.group(1))
        imperial = m.group(2).startswith(("lb", "pound"))
        found.append((m.start(), value * LB_TO_KG if imperial else value, imperial))
        spans.append(m.start(1))
    for m in _WEIGHT_CUE.finditer(t):
        if m.start(1) in spans:
            continue
        value = float(m.group(1))
        imperial = imperial_hint or value > 150
        found.append((m.start(), value * LB_TO_KG if imperial else value, imperial))
    delta_spans = [(d.start(2), d.end(2)) for d in _DELTA.finditer(t)]
    current = target = None
    imperial_used = False
    for pos, kg, imperial in sorted(found):
        if any(a - 12 <= pos <= b + 8 for a, b in delta_spans):
            continue
        if not _in_range("weight_kg", kg):
            continue
        imperial_used = imperial_used or imperial
        before = t[max(0, pos - 30):pos]
        if _TARGET_CUE.search(before):
            target = target if target is not None else kg
        else:
            current = current if current is not None else kg
    return current, target, imperial_used


def extract_height(t: str) -> tuple[Optional[float], bool]:
    m = _FEET_WORD.search(t) or _FEET_TICK.search(t)
    if m:
        feet, inches = float(m.group(1)), float(m.group(2) or 0)
        if inches < 12:
            return (feet * 12 + inches) * 2.54, True
    for rx in (_CM, _HEIGHT_CUE):
        m = rx.search(t)
        if m:
            return float(m.group(1)), False
    m = _METRES.search(t)
    if m:
        return float(m.group(1)) * 100, False
    return None, False


def extract_age(t: str) -> tuple[Optional[int], bool]:
    """Return (age, explicit). Looser patterns such as '2 years' are never reported as invalid."""
    for i, rx in enumerate(_AGE_PATTERNS):
        m = rx.search(t)
        if m:
            return int(m.group(1)), i != 1
    return None, False


def extract(text: str, expecting: Optional[list[str]] = None, units: str = "metric") -> dict:
    """Pull every recognisable slot from `text`. Keys are only present when found."""
    t = clean(text)
    out: dict = {}
    invalid: list[str] = []

    age, explicit_age = extract_age(t)
    if age is not None:
        if _in_range("age", age):
            out["age"] = age
        elif explicit_age:
            invalid.append("age")

    if _SEX_OTHER.search(t):
        out["sex"] = "other"
    elif _SEX_FEMALE.search(t):
        out["sex"] = "female"
    elif _SEX_MALE.search(t):
        out["sex"] = "male"

    height, height_imperial = extract_height(t)
    if height is not None:
        (out.__setitem__("height_cm", round(height, 1)) if _in_range("height_cm", height) else invalid.append("height_cm"))
    current, target, weight_imperial = extract_weights(t, units == "imperial")
    if current is not None:
        out["weight_kg"] = round(current, 1)
    if target is not None:
        out["goal_weight_kg"] = round(target, 1)
    if height_imperial or weight_imperial:
        out["units"] = "imperial"
    elif re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|kgs|cm|cms)\b", t):
        out["units"] = "metric"

    delta = _DELTA.search(t)
    if delta:
        amount = float(delta.group(2))
        if delta.group(3).startswith(("lb", "pound")):
            amount *= LB_TO_KG
        sign = 1 if delta.group(1) in {"gain", "put on"} else -1
        out["goal_delta_kg"] = round(sign * amount, 1)

    frame = _TIMEFRAME.search(t)
    if frame and "goal_delta_kg" in out:
        count, unit = int(frame.group(1)), frame.group(2)
        out["timeframe_weeks"] = round(count / 7 if unit.startswith("day") else count * 4.345 if unit.startswith("month") else count, 1)

    gain, lose, hold = _GOAL_GAIN.search(t), _GOAL_LOSE.search(t), _GOAL_MAINTAIN.search(t)
    if gain and lose:
        out["goal"] = "maintain"
    elif gain:
        out["goal"] = "gain"
    elif lose:
        out["goal"] = "lose"
    elif hold:
        out["goal"] = "maintain"
    elif "goal_delta_kg" in out:
        out["goal"] = "gain" if out["goal_delta_kg"] > 0 else "lose"

    days = None
    span = re.search(r"\b([1-7])\s*(?:-|to)\s*([1-7])\s*(?:days?|x|times)\b", t)
    if span:
        days = round((int(span.group(1)) + int(span.group(2))) / 2)
    for rx in ([] if days else _DAYS):
        m = rx.search(t)
        if m:
            raw = m.group(1)
            days = WORD_NUMBERS.get(raw, int(raw) if raw.isdigit() else None)
            break
    if days is not None:
        out["days"] = days
    for level, rx in _ACTIVITY:
        if rx.search(t):
            out["activity"] = level
            break

    for diet, rx in _DIETS:
        if rx.search(t):
            out["diet"] = diet
            break
    if _CLEAR_ALLERGENS.search(t):
        out["allergens"] = []
    else:
        hits = [name for name, rx in _ALLERGENS if rx.search(t)]
        if hits:
            out["allergens"] = hits

    m = _MEALS.search(t)
    if m:
        raw = m.group(1) or m.group(2)
        out["meals"] = int(raw) if raw.isdigit() else {"three": 3, "four": 4, "five": 5}[raw]
    m = _PACE.search(t)
    if m:
        fast = m.group(1) or m.group(3)
        out["pace"] = "fast" if fast else "gentle" if (m.group(2) or m.group(4)) in {"slow", "gentle", "slowly", "gently", "easy"} else "steady"
    for name, rx in _EQUIPMENT:
        if rx.search(t):
            out["equipment"] = name
            break
    for name, rx in _EXPERIENCE:
        if rx.search(t):
            out["experience"] = name
            break
    for name, rx in _FOCUS:
        if rx.search(t):
            out["focus"] = name
            break

    if expecting:
        nums = re.findall(r"(?<![\d.])\d{1,3}(?:\.\d+)?(?![\d.])", t)
        bare = re.fullmatch(r"[\d.\s]+(?:kg|kgs|lb|lbs|cm|cms|years?|yrs?|yo)?", t) is not None
        if bare and nums and not any(k in out for k in ("age", "height_cm", "weight_kg")):
            values = [float(n) for n in nums[:2]]
            unit_lb = re.search(r"\b(lb|lbs)\b", t) is not None
            remaining = [f for f in expecting if f in RANGES]
            for value in values:
                for field in list(remaining):
                    v = value * LB_TO_KG if (field == "weight_kg" and unit_lb) else value
                    if _in_range(field, v):
                        out[field] = int(v) if field == "age" else round(v, 1)
                        remaining.remove(field)
                        break
    if invalid:
        out["invalid"] = invalid
    return out
