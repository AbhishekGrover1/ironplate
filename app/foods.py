"""Food database: search index, portion parser and food recommendations.

Search combines exact alias lookup, phrase containment and a character n-gram
TF-IDF index, so typos and plurals still land on the right food. The portion
parser turns free text such as "2 rotis and a bowl of dal" into grams.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DATA_DIR

DIET_LEVEL = {"vegan": 0, "vegetarian": 1, "eggetarian": 2, "pescatarian": 3, "nonveg": 4}
MEALS = ("breakfast", "lunch", "dinner", "snack")
MATCH_MIN = 0.5

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75, "half": 0.5, "quarter": 0.25}
GLYPHS = {"¼", "½", "¾"}

UNIT_ALIASES = {
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g", "gr": "g",
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "ml": "ml", "l": "l", "litre": "l", "liter": "l", "litres": "l", "liters": "l",
    "cup": "cup", "cups": "cup",
    "bowl": "bowl", "bowls": "bowl",
    "katori": "katori", "katoris": "katori", "katora": "katori",
    "tbsp": "tbsp", "tbs": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "slice": "slice", "slices": "slice",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece",
    "glass": "glass", "glasses": "glass",
    "scoop": "scoop", "scoops": "scoop",
    "plate": "plate", "plates": "plate",
    "serving": "serving", "servings": "serving", "portion": "serving", "portions": "serving",
    "handful": "handful", "handfuls": "handful",
    "can": "can", "cans": "can", "bar": "bar", "bars": "bar",
}

# Units that read naturally without a gram figure next to them.
COUNTABLE_UNITS = {
    "egg", "white", "roti", "slice", "banana", "apple", "orange", "date", "idli", "dosa",
    "bar", "scoop", "tbsp", "tsp", "paratha", "naan", "biscuit", "samosa", "burger",
}

LEAD_FILLER = {
    "i", "ive", "i've", "we", "just", "also", "then", "and", "had", "have", "has", "ate",
    "eat", "eaten", "eating", "consumed", "drank", "drink", "drinking", "log", "logged",
    "add", "added", "track", "tracked", "record", "recorded", "took", "take", "finished",
    "enjoyed", "having", "got", "my", "please", "today", "tonight", "earlier", "now",
    "this", "morning", "afternoon", "evening", "night", "was", "were", "is", "are", "it",
    "that", "some", "about", "around", "approx", "approximately", "roughly", "only", "to",
    "in", "on", "the", "of", "me", "for", "at", "as",
}
DESCRIPTORS = {
    "fresh", "homemade", "home", "made", "organic", "small", "large", "medium", "big",
    "little", "extra", "hot", "cold", "tasty", "healthy", "plain",
}
QUESTION_WORDS = {
    "how", "many", "much", "calories", "calorie", "cals", "cal", "kcal", "protein",
    "proteins", "carbs", "carb", "carbohydrates", "carbohydrate", "fat", "fats", "fiber",
    "fibre", "macros", "macro", "nutrition", "nutritional", "nutrients", "facts", "info",
    "information", "value", "values", "content", "count", "in", "of", "for", "does", "do",
    "did", "has", "have", "is", "are", "was", "the", "tell", "me", "about", "what",
    "whats", "what's", "show", "give", "check", "per", "good", "bad", "healthy",
    "contain", "contains", "there", "its", "it's", "it", "i", "my", "eat", "eating",
    "a", "an", "high", "low", "weight", "loss", "gain", "fattening", "worth", "should",
    "can", "get", "from", "to", "on", "and", "nutritive", "stats", "details", "total",
}

NUTRIENT_WORDS = {"protein", "proteins", "carbs", "carb", "carbohydrates", "carbohydrate", "fat", "fats", "fiber", "fibre"}
OIL_DENSITY = 0.92


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    text = text.lower().replace("’", "'").replace("&", " and ")
    text = re.sub(r"[^a-z0-9%¼½¾/'.\-| ]", " ", text)
    text = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem(word: str) -> str:
    """A tiny, consistent stemmer so eggs/egg and tomatoes/tomato agree."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith(("ss", "us")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def norm_phrase(text: str) -> str:
    return " ".join(stem(w) for w in re.sub(r"[^a-z0-9% ]", " ", normalize_text(text)).split())


def fmt_count(x: float) -> str:
    whole, frac = int(x), round((x - int(x)) * 4) / 4
    glyph = {0.0: "", 0.25: "¼", 0.5: "½", 0.75: "¾", 1.0: ""}[frac]
    if frac == 1.0:
        whole += 1
    if whole == 0 and glyph:
        return glyph
    return f"{whole}{glyph}"


def plural(unit: str, n: float) -> str:
    if n <= 1.0001 or unit in {"tbsp", "tsp"}:
        return unit
    if unit.endswith(("s", "x", "ch", "sh")):
        return unit + "es"
    return unit + "s"


@dataclass(frozen=True)
class Food:
    id: int
    name: str
    aliases: tuple[str, ...]
    role: str
    diet: str
    allergens: frozenset[str]
    slots: tuple[str, ...]
    kcal: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    unit: str
    unit_g: float
    cup_g: Optional[float]
    bowl_g: Optional[float]
    step_g: float
    max_steps: int
    display: str

    @property
    def label(self) -> str:
        return self.name[:1].upper() + self.name[1:]

    def nutrients(self, grams: float) -> dict[str, float]:
        k = grams / 100.0
        return {
            "kcal": self.kcal * k, "protein": self.protein * k, "carbs": self.carbs * k,
            "fat": self.fat * k, "fiber": self.fiber * k,
        }

    def portion_label(self, grams: float) -> str:
        """Human portion text, e.g. '2 eggs', '1½ cups (237 g)' or '150 g'."""
        shown = round(grams)
        if self.display == "unit" and self.unit_g > 0:
            n = grams / self.unit_g
            q = round(n * 4) / 4
            if q > 0 and abs(n - q) <= 0.04 * max(q, 1):
                label = f"{fmt_count(q)} {plural(self.unit, q)}"
                return label if self.unit in COUNTABLE_UNITS else f"{label} ({shown} g)"
        return f"{shown} g"

    def summary(self, grams: float, portion: Optional[str] = None) -> dict:
        n = self.nutrients(grams)
        return {
            "id": self.id, "name": self.label, "grams": round(grams),
            "portion": portion or self.portion_label(grams), "kcal": round(n["kcal"]),
            "protein": round(n["protein"], 1), "carbs": round(n["carbs"], 1),
            "fat": round(n["fat"], 1), "fiber": round(n["fiber"], 1),
        }


@dataclass
class ParsedItem:
    food: Food
    grams: float
    portion: str
    query: str
    score: float

    def summary(self) -> dict:
        data = self.food.summary(self.grams, self.portion)
        data["query"] = self.query
        data["exact"] = self.score >= 0.95
        return data


@dataclass
class ParsedMeal:
    meal: Optional[str]
    items: list[ParsedItem]
    unmatched: list[str]


def _blank(v: str) -> Optional[float]:
    return float(v) if v.strip() else None


class FoodDB:
    """In-memory food table with a fuzzy search index."""

    def __init__(self, foods: list[Food]):
        self.foods = foods
        self.by_id = {f.id: f for f in foods}
        variants: list[str] = []
        owners: list[int] = []
        self._exact: dict[str, Food] = {}
        self._phrases: list[tuple[str, Food]] = []
        for food in foods:
            for text in (food.name, *food.aliases):
                phrase = norm_phrase(text)
                if phrase and phrase not in self._exact:
                    self._exact[phrase] = food
                    self._phrases.append((phrase, food))
        for food in foods:
            bare = norm_phrase(re.sub(r"\s*\(.*?\)", "", food.name))
            if bare and bare not in self._exact:
                self._exact[bare] = food
                self._phrases.append((bare, food))
        for phrase, food in self._phrases:
            variants.append(phrase)
            owners.append(food.id)
        self._owners = np.array(owners)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self._matrix = self._vec.fit_transform(variants)
        self._protected = [p for p, _ in self._phrases if " with " in f" {p} " or " and " in f" {p} "]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FoodDB":
        path = path or DATA_DIR / "foods.csv"
        foods: list[Food] = []
        with open(path, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                foods.append(Food(
                    id=i, name=row["name"].strip().lower(),
                    aliases=tuple(a.strip().lower() for a in row["aliases"].split("|") if a.strip()),
                    role=row["role"], diet=row["diet"],
                    allergens=frozenset(a for a in row["allergens"].split("|") if a),
                    slots=tuple(s for s in row["slots"].split("|") if s),
                    kcal=float(row["kcal"]), protein=float(row["protein"]),
                    carbs=float(row["carbs"]), fat=float(row["fat"]), fiber=float(row["fiber"]),
                    unit=row["unit"], unit_g=float(row["unit_g"]),
                    cup_g=_blank(row["cup_g"]), bowl_g=_blank(row["bowl_g"]),
                    step_g=float(row["step_g"]), max_steps=int(row["max_steps"]),
                    display=row["display"],
                ))
        return cls(foods)

    # ------------------------------------------------------------------- search
    def _score(self, query: str) -> dict[int, float]:
        qn = norm_phrase(query)
        scores: dict[int, float] = {}
        if not qn:
            return scores
        if qn in self._exact:
            scores[self._exact[qn].id] = 1.0
        padded = f" {qn} "
        for phrase, food in self._phrases:
            if len(phrase) >= 3 and f" {phrase} " in padded:
                s = 0.6 + 0.4 * len(phrase) / len(qn)
                scores[food.id] = max(scores.get(food.id, 0.0), min(s, 0.99))
        sims = (self._matrix @ self._vec.transform([qn]).T).toarray().ravel()
        for idx in np.argsort(-sims)[:12]:
            fid = int(self._owners[idx])
            scores[fid] = max(scores.get(fid, 0.0), float(sims[idx]))
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[Food, float]]:
        scores = self._score(query)
        stripped = " ".join(w for w in normalize_text(query).split() if w not in DESCRIPTORS)
        if stripped and stripped != normalize_text(query) and max(scores.values(), default=0) < 0.8:
            for fid, s in self._score(stripped).items():
                scores[fid] = max(scores.get(fid, 0.0), s)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], len(self.by_id[kv[0]].name)))
        return [(self.by_id[fid], s) for fid, s in ranked[:k]]

    def match(self, query: str) -> Optional[tuple[Food, float]]:
        hits = self.search(query, 1)
        if hits and hits[0][1] >= MATCH_MIN:
            return hits[0]
        return None

    # ----------------------------------------------------------------- portions
    def to_grams(self, food: Food, qty: Optional[float], unit: Optional[str]) -> float:
        q = 1.0 if qty is None else qty
        density = OIL_DENSITY if food.role == "fat" and food.fat > 90 else 1.0
        if unit is None or unit == "serving":
            return q * food.unit_g
        if unit == "g":
            return q
        if unit == "kg":
            return q * 1000
        if unit == "oz":
            return q * 28.35
        if unit == "lb":
            return q * 453.6
        if unit == "ml":
            return q * density
        if unit == "l":
            return q * 1000 * density
        cup = food.cup_g or (food.unit_g if food.unit == "cup" else 150.0)
        if unit == "cup":
            return q * cup
        if unit == "bowl":
            return q * (food.bowl_g or food.cup_g or (food.unit_g if food.unit == "bowl" else 150.0))
        if unit == "katori":
            return q * (food.unit_g if food.unit == "katori" else 0.6 * (food.cup_g or 250.0))
        if unit in {"tbsp", "tsp"}:
            if food.unit == "tbsp":
                tbsp = food.unit_g
            elif food.unit == "tsp":
                tbsp = food.unit_g * 3
            else:
                tbsp = (food.cup_g / 16) if food.cup_g else 15.0 * density
            return q * (tbsp if unit == "tbsp" else tbsp / 3)
        if unit in {"slice", "piece", "bar", "scoop", "handful", "glass", "can", "plate"}:
            if food.unit == unit:
                return q * food.unit_g
            fallback = {"glass": food.cup_g or 240.0, "scoop": 60.0, "handful": 30.0,
                        "can": 330.0, "bar": 60.0, "plate": (food.cup_g * 2) if food.cup_g else food.unit_g}
            return q * fallback.get(unit, food.unit_g)
        return q * food.unit_g

    @staticmethod
    def _take_quantity(tokens: list[str]) -> tuple[Optional[float], list[str]]:
        if not tokens:
            return None, tokens
        tok, rest = tokens[0], tokens[1:]
        qty: Optional[float] = None
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            qty = float(tok)
            if rest and re.fullmatch(r"\d+/\d+", rest[0]):
                a, b = rest[0].split("/")
                qty += float(a) / float(b) if float(b) else 0.0
                rest = rest[1:]
            elif rest and rest[0] in GLYPHS:
                qty += FRACTIONS[rest[0]]
                rest = rest[1:]
        elif re.fullmatch(r"\d+/\d+", tok):
            a, b = tok.split("/")
            qty = float(a) / float(b) if float(b) else None
        elif tok in FRACTIONS:
            qty = FRACTIONS[tok]
            if rest and rest[0] in {"a", "an"}:
                rest = rest[1:]
        elif tok == "a" and rest[:1] == ["couple"]:
            qty, rest = 2.0, rest[1:]
            if rest[:1] == ["of"]:
                rest = rest[1:]
        elif tok == "couple":
            qty = 2.0
            if rest[:1] == ["of"]:
                rest = rest[1:]
        elif tok in NUMBER_WORDS:
            qty = float(NUMBER_WORDS[tok])
        if qty is None:
            return None, tokens
        if rest and rest[0] in {"x", "×"}:
            rest = rest[1:]
        if rest and rest[0] == "dozen":
            qty *= 12
            rest = rest[1:]
        return qty, rest

    @staticmethod
    def _split_on_quantity(tokens: list[str]) -> list[str]:
        """'2 eggs 2 slices of toast' -> ['2 eggs', '2 slices of toast']."""
        segments: list[str] = []
        current: list[str] = []
        has_word = False
        for tok in tokens:
            is_num = bool(re.fullmatch(r"\d+(?:\.\d+)?", tok))
            if is_num and has_word:
                segments.append(" ".join(current))
                current, has_word = [], False
            current.append(tok)
            plain = (is_num or tok in FRACTIONS or tok in UNIT_ALIASES or tok in NUMBER_WORDS
                     or tok in {"of", "x", "×", "dozen", "couple"} or re.fullmatch(r"\d+/\d+", tok))
            if not plain:
                has_word = True
        if current:
            segments.append(" ".join(current))
        return segments

    def _best(self, query: str) -> float:
        return max(self._score(query).values(), default=0.0)

    def _split_foods(self, query: str) -> list[str]:
        """'rotis dal' -> ['rotis', 'dal'] when every word is an exact food name."""
        toks = query.split()
        if len(toks) < 2 or self._best(query) >= 0.95:
            return [query]
        parts: list[str] = []
        i = 0
        while i < len(toks):
            for j in range(len(toks), i, -1):
                if norm_phrase(" ".join(toks[i:j])) in self._exact:
                    parts.append(" ".join(toks[i:j]))
                    i = j
                    break
            else:
                return [query]
        return parts if len(parts) > 1 else [query]

    def parse_meal(self, text: str, hour: Optional[int] = None) -> ParsedMeal:
        """Turn 'for lunch I had 2 rotis, dal and a bowl of rice' into foods and grams."""
        t = re.sub(r"[,;\n+]", " | ", text.lower())
        t = normalize_text(t)
        t = re.sub(r"(\d)\s*(?:and a half|and half)\b", r"\1.5", t)
        t = re.sub(r"\bone and a half\b", "1.5", t)
        t = re.sub(r"(?<=\d)(?=[a-z])", " ", t)
        meal: Optional[str] = None
        m = re.search(r"\b(breakfast|brunch|lunch|dinner|supper|snacks?)\b", t)
        if m:
            meal = {"brunch": "breakfast", "supper": "dinner", "snacks": "snack"}.get(m.group(1), m.group(1))
            t = re.sub(r"\b(?:for|at|as|during|in|to)?\s*(?:my\s+|the\s+)?"
                       r"(?:breakfast|brunch|lunch|dinner|supper|snacks?)\b", " ", t)
        t = re.sub(r"\b(?:to|in|into|on)\s+(?:my\s+)?(?:food\s+)?(?:log|diary|tracker|journal)\b", " ", t)
        t = re.sub(r"\b(?:today|tonight|just now|earlier|this morning|this evening)\b", " ", t)
        if meal is None and hour is not None:
            meal = ("breakfast" if hour < 11 else "lunch" if hour < 15 else "snack" if hour < 18 else "dinner")
        for phrase in self._protected:
            if f" {phrase} " in f" {t} ":
                t = re.sub(rf"\b{re.escape(phrase)}\b", phrase.replace(" ", "_"), t)
        tokens = t.split()
        while tokens and tokens[0] in LEAD_FILLER:
            tokens.pop(0)
        t = " ".join(tokens)
        segments: list[str] = []
        for part in re.split(r"\||\band\b|\bwith\b|\bplus\b|\bthen\b|\balong\b", t):
            segments.extend(self._split_on_quantity(part.replace("_", " ").split()))
        items: list[ParsedItem] = []
        unmatched: list[str] = []
        for seg in segments:
            toks = [w for w in seg.split() if w not in {"i", "had", "ate", "have", "some", "my"}]
            qty, rest = self._take_quantity(toks)
            unit: Optional[str] = None
            if rest and rest[0] in UNIT_ALIASES and not (len(rest) == 1 and qty is None):
                unit = UNIT_ALIASES[rest[0]]
                rest = rest[1:]
                if rest and rest[0] == "of":
                    rest = rest[1:]
            query = " ".join(w for w in rest if w not in {"of", "the"}).strip()
            if not query:
                continue
            for n, sub in enumerate(self._split_foods(query)):
                hit = self.match(sub)
                if hit is None:
                    unmatched.append(sub)
                    continue
                food, score = hit
                q, u = (qty, unit) if n == 0 else (None, None)
                grams = max(1.0, min(self.to_grams(food, q, u), 3000.0))
                if u and u not in {"g", "kg", "oz", "lb", "ml", "l"}:
                    c = 1.0 if q is None else q
                    portion = f"{fmt_count(c)} {plural(u, c)} ({round(grams)} g)"
                elif u:
                    portion = f"{round(grams)} g"
                else:
                    portion = food.portion_label(grams)
                items.append(ParsedItem(food, grams, portion, sub, score))
        return ParsedMeal(meal, items, unmatched)

    @staticmethod
    def strip_question_words(text: str, keep_nutrients: bool = False) -> str:
        skip = QUESTION_WORDS - NUTRIENT_WORDS if keep_nutrients else QUESTION_WORDS
        return " ".join(w for w in normalize_text(text).split() if w not in skip)

    # ------------------------------------------------------------ recommendations
    def allowed(self, diet: Optional[str] = None, allergens: Iterable[str] = (),
                avoid: Iterable[str] = (), slot: Optional[str] = None,
                plannable_only: bool = True) -> list[Food]:
        level = DIET_LEVEL.get(diet or "nonveg", 4)
        banned = set(allergens)
        avoid_ids = {int(a) for a in avoid if str(a).isdigit()}
        out = []
        for f in self.foods:
            if DIET_LEVEL[f.diet] > level or f.allergens & banned or f.id in avoid_ids:
                continue
            if plannable_only and not f.slots:
                continue
            if slot and slot not in f.slots:
                continue
            out.append(f)
        return out

    def rank(self, metric: str, pool: list[Food], k: int = 8) -> list[Food]:
        """Rank foods for 'high protein', 'high fiber', 'low calorie' style requests."""
        def per_serving(f: Food) -> dict[str, float]:
            return f.nutrients(f.unit_g)

        scored: list[tuple[float, Food]] = []
        for f in pool:
            s = per_serving(f)
            if metric == "protein" and s["protein"] >= 6 and f.kcal > 0:
                scored.append((f.protein / f.kcal, f))
            elif metric == "fiber" and s["fiber"] >= 3 and f.kcal > 0:
                scored.append((f.fiber / f.kcal, f))
            elif metric == "lowcal" and s["kcal"] >= 25:
                scored.append((-s["kcal"], f))
            elif metric == "lowcarb" and s["kcal"] >= 60 and s["protein"] + s["fat"] >= 6:
                scored.append((-f.carbs / max(f.kcal, 1), f))
            elif metric == "lowfat" and s["kcal"] >= 40 and f.role != "fat":
                scored.append((-f.fat / max(f.kcal, 1), f))
            elif metric == "healthy" and s["kcal"] >= 40 and f.role != "fat":
                scored.append(((f.protein + 2 * f.fiber) / max(f.kcal, 1), f))
            elif metric == "healthyfat" and f.name in HEALTHY_FATS:
                scored.append((f.fat / max(f.kcal, 1), f))
        scored.sort(key=lambda p: -p[0])
        return [f for _, f in scored[:k]]

    def similar(self, food: Food, pool: list[Food], k: int = 5) -> list[tuple[Food, float]]:
        """Nearest neighbours in nutrient space within the same role (for swaps)."""
        cols = np.array([[f.kcal, f.protein, f.carbs, f.fat, f.fiber] for f in self.foods])
        mean, std = cols.mean(axis=0), cols.std(axis=0) + 1e-9
        weights = np.array([1.0, 1.0, 1.5, 1.0, 0.6])

        def vec(f: Food) -> np.ndarray:
            return (np.array([f.kcal, f.protein, f.carbs, f.fat, f.fiber]) - mean) / std * weights

        base = vec(food)
        out = []
        for f in pool:
            if f.id == food.id or f.role != food.role:
                continue
            out.append((f, float(np.linalg.norm(vec(f) - base))))
        out.sort(key=lambda p: p[1])
        return out[:k]


HEALTHY_FATS = {
    "almonds", "walnuts", "cashews", "peanuts (roasted)", "avocado", "olive oil",
    "chia seeds", "flaxseeds", "peanut butter", "hummus",
}
