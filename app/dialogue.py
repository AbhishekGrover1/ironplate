"""Conversation manager.

Turns one message plus the browser's saved state into a reply, structured blocks for
the interface and state updates. Order of work: reset confirmation, safety checks,
FAQ, slot extraction, intent classification with rule-based corrections, handler.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .entities import extract
from .faq import Faq
from .foods import MEALS, Food, FoodDB, stem
from .nlu import CONFIDENCE_MIN, IntentModel
from .nutrition import ACTIVITY_LABEL, Targets, compute_targets, missing_fields
from .planner import plan_day
from .safety import check as safety_check
from .schemas import ClientState
from .trend import analyze, clean_series, parse_day
from .training import ensure_model
from .workouts import EQUIPMENT_LABEL, ExerciseDB, build_plan

CORE = ("age", "sex", "height_cm", "weight_kg")
FIELD_NAMES = {"age": "age", "sex": "sex", "height_cm": "height", "weight_kg": "weight"}
PROFILE_KEYS = ("age", "sex", "height_cm", "weight_kg", "goal", "goal_weight_kg", "activity", "diet",
                "allergens", "meals", "pace", "equipment", "experience", "focus", "days", "units")
PROFILE_INTENTS = {"set_profile", "calc_targets", "meal_plan", "workout_plan", "log_weight", "progress"}
GOAL_TEXT = {"lose": "lose fat", "gain": "build muscle", "maintain": "maintain weight"}
DIET_TEXT = {"vegan": "vegan", "vegetarian": "vegetarian", "eggetarian": "eggetarian",
             "pescatarian": "pescatarian", "nonveg": "non-vegetarian"}
LB = 0.45359237
ALLERGEN_TEXT = {"nuts": "tree nuts", "peanut": "peanuts", "egg": "eggs", "shellfish": "shellfish"}

LOG_VERBS = re.compile(r"\b(ate|had|have|eaten|log|logged|add|added|track|tracked|consumed|drank|finished|eating)\b")
AGAIN = re.compile(r"\b(another|different|new|again|regenerate|more|other|else|redo|change)\b")
YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|confirm|do it|go ahead|delete)\b")
NO = re.compile(r"^(?:no|nope|cancel|stop|don'?t|never ?mind)\b")
AVOID = re.compile(r"\b(?:no|without|avoid|exclude|skip|hate|dislike|don'?t (?:like|eat|want)|do not (?:like|eat|want))\s+"
                   r"([a-z][a-z ]{2,30}?)(?=$|[.,;!?]|\band\b|\bbut\b|\bplease\b)")
NOT_FOODS = {"dairy", "gluten", "nuts", "nut", "peanut", "peanuts", "soy", "eggs", "fish", "shellfish",
             "sesame", "meat", "sugar", "carbs", "carb", "fat", "oil", "salt", "lactose", "equipment", "weights"}


def n(x: float) -> str:
    return f"{round(x):,}"


def join_words(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def is_complete(profile: dict) -> bool:
    return not missing_fields(profile)


def food_label(name: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", name).strip().lower()


def log_phrase(item: dict) -> str:
    """'2 rotis', '1 banana' or '158 g white rice': always something the parser reads back correctly."""
    label = food_label(item["name"])
    portion = item["portion"]
    if "(" in portion:
        return f"{item['grams']} g {label}"
    if any(stem(w) in stem(portion) for w in label.split()):
        return portion
    return f"{portion} {label}"


@dataclass
class Turn:
    text: str
    state: ClientState
    profile: dict
    plan_seed: int = 0
    intent: str = "fallback"
    confidence: float = 0.0
    entities: dict = field(default_factory=dict)
    pending: Optional[dict] = None
    lines: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    chips: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def say(self, *lines: str) -> None:
        self.lines.extend(line for line in lines if line)


class Assistant:
    """Loads the data and models once and answers messages."""

    def __init__(self, foods: FoodDB, exercises: ExerciseDB, model: IntentModel, faq: Faq):
        self.foods, self.exercises, self.model, self.faq = foods, exercises, model, faq
        self._handlers = {
            "greet": self._on_greet, "thanks": self._on_thanks, "bye": self._on_bye, "help": self._on_help,
            "about": self._on_about, "calc_targets": self._on_calc_targets, "meal_plan": self._on_meal_plan,
            "workout_plan": self._on_workout_plan, "log_food": self._on_log_food, "food_info": self._on_food_info,
            "food_suggest": self._on_food_suggest, "log_weight": self._on_log_weight,
            "progress": self._on_progress, "today_summary": self._on_today_summary,
            "exercise_info": self._on_exercise_info, "reset": self._on_reset,
        }

    @classmethod
    def load(cls) -> "Assistant":
        return cls(FoodDB.load(), ExerciseDB.load(), ensure_model(), Faq.load())

    # ------------------------------------------------------------------ entry
    def respond(self, message: str, state: ClientState) -> dict:
        text = re.sub(r"\s+", " ", message).strip()
        t = Turn(text=text, state=state, profile=dict(state.profile), plan_seed=state.plan_seed)

        if state.pending and state.pending.get("type") == "confirm_reset":
            answer = text.lower()
            if YES.match(answer):
                t.intent, t.profile, t.plan_seed = "reset", {}, 0
                t.actions.append({"type": "reset_all"})
                t.say("Done. Your profile, food log and weights are cleared.")
                t.chips = self._starter_chips(t)
                return self._finish(t)
            if NO.match(answer):
                t.intent = "reset"
                t.say("Okay, nothing was deleted.")
                t.chips = self._starter_chips(t)
                return self._finish(t)

        hit = safety_check(text)
        if hit:
            t.intent, t.confidence = f"safety_{hit[0]}", 1.0
            t.say(hit[1])
            if hit[0] == "pregnancy":
                t.chips = ["I had 2 eggs for breakfast", "What's left today"]
            return self._finish(t)

        waiting = bool(state.pending and state.pending.get("type") == "need_profile")
        expecting = state.pending.get("missing") if waiting else None
        t.entities = extract(text, expecting=expecting, units=t.profile.get("units", "metric"))
        measured = any(k in t.entities for k in ("age", "height_cm", "weight_kg", "goal_delta_kg"))
        if not waiting and not measured:
            answer = self.faq.lookup(text)
            if answer:
                t.intent, t.confidence = "faq", 1.0
                t.say(answer[0])
                t.chips = self._starter_chips(t)
                return self._finish(t)
        intent, conf, _ = self.model.predict(text)
        intent, conf, then = self._correct(t, intent, conf, waiting)
        t.intent, t.confidence = intent, conf

        if intent in PROFILE_INTENTS:
            self._apply(t)
        if intent in {"set_profile", "meal_plan"}:
            self._remember_dislikes(t)
        if intent == "set_profile":
            self._on_set_profile(t, then)
        else:
            self._handlers.get(intent, self._on_fallback)(t)
        return self._finish(t)

    def _finish(self, t: Turn) -> dict:
        targets = self._targets(t)
        return {
            "reply": "\n".join(t.lines), "blocks": t.blocks, "profile": t.profile, "pending": t.pending,
            "plan_seed": t.plan_seed, "targets": targets.to_dict() if targets else None,
            "actions": t.actions, "chips": t.chips[:4], "intent": t.intent, "confidence": round(t.confidence, 2),
        }

    # ------------------------------------------------------------ understanding
    def _correct(self, t: Turn, intent: str, conf: float, waiting: bool) -> tuple[str, float, Optional[str]]:
        """Rule-based corrections on top of the classifier."""
        e = t.entities
        found = [k for k in PROFILE_KEYS if k in e and k != "units"]
        if waiting and found:
            return "set_profile", 1.0, (t.state.pending or {}).get("then")
        core = [k for k in CORE if k in e]
        if len(core) >= 2 and intent not in {"set_profile", "calc_targets", "meal_plan", "workout_plan"}:
            return "set_profile", max(conf, 0.6), None
        low = t.text.lower()
        if (intent not in {"meal_plan", "food_suggest", "workout_plan", "calc_targets"}
                and not LOG_VERBS.search(low) and self._dislikes(t.text)):
            return "set_profile", max(conf, 0.7), None
        if conf < CONFIDENCE_MIN or intent in {"out_of_scope", "log_food", "food_info"}:
            parsed = self.foods.parse_meal(t.text, t.state.hour)
            all_exact = bool(parsed.items) and not parsed.unmatched and all(i.score >= 0.95 for i in parsed.items)
            has_qty = bool(re.search(r"\d|\b(?:a|an|one|two|three|four|half|couple)\b", low))
            verb = bool(LOG_VERBS.search(low))
            if intent == "log_food" and len(parsed.items) == 1 and not verb and not has_qty:
                return "food_info", conf, None
            if conf < CONFIDENCE_MIN or intent == "out_of_scope":
                if found:
                    return "set_profile", 0.5, None
                if all_exact:
                    return ("log_food" if verb or has_qty else "food_info"), 0.5, None
                if conf < CONFIDENCE_MIN:
                    return "fallback", conf, None
        return intent, conf, None

    def _describe(self, key: str, value, profile: dict) -> str:
        imperial = profile.get("units") == "imperial"
        if key == "age":
            return f"age {value}"
        if key == "sex":
            return str(value)
        if key == "height_cm":
            if imperial:
                feet, inch = divmod(round(value / 2.54), 12)
                return f"{feet}'{inch}\" ({round(value)} cm)"
            return f"{round(value)} cm"
        if key in {"weight_kg", "goal_weight_kg"}:
            label = "goal weight " if key == "goal_weight_kg" else ""
            return f"{label}{value:g} kg" + (f" ({round(value / LB)} lb)" if imperial else "")
        mapping = {
            "goal": lambda v: "goal: " + GOAL_TEXT[v], "activity": lambda v: ACTIVITY_LABEL[v],
            "diet": lambda v: DIET_TEXT[v], "days": lambda v: f"{v} training days a week",
            "equipment": lambda v: "training: " + EQUIPMENT_LABEL[v], "experience": str,
            "focus": lambda v: {"strength": "strength focus", "hypertrophy": "muscle focus", "general": "general fitness"}[v],
            "pace": lambda v: f"{v} pace", "meals": lambda v: f"{v} meals a day",
        }
        return mapping.get(key, lambda v: f"{key} {v}")(value)

    def _apply(self, t: Turn) -> None:
        e, p = t.entities, t.profile
        for key in PROFILE_KEYS:
            if key not in e:
                continue
            value = e[key]
            if key == "allergens":
                current = set(p.get("allergens", []))
                new = [] if value == [] else sorted(current | set(value))
                if new != sorted(current):
                    p["allergens"] = new
                    t.changed.append(("no " + join_words(ALLERGEN_TEXT.get(a, a) for a in new)) if new else "restrictions cleared")
                continue
            if key == "goal_weight_kg":
                self._set_goal_weight(t, value)
                continue
            if key == "weight_kg":
                t.actions.append({"type": "log_weight", "date": t.state.today, "kg": value})
            if p.get(key) != value:
                p[key] = value
                if key != "units":
                    t.changed.append(self._describe(key, value, p))
        delta = e.get("goal_delta_kg")
        if delta is not None:
            weight = p.get("weight_kg")
            weeks = e.get("timeframe_weeks")
            limit = (0.01 if delta < 0 else 0.005) * (weight or 75.0)
            if weeks and abs(delta) / max(weeks, 0.5) > max(limit, 0.25):
                weekly = abs(delta) / max(weeks, 0.5)
                t.notes.append(
                    f"{abs(delta):g} kg in {weeks:g} weeks is about {weekly:.1f} kg a week, which is faster than is "
                    f"realistic or safe for most people. About {limit:.1f} kg a week is the upper end for you, so I "
                    f"will keep your calorie target inside that range.")
                p["pace"] = "steady"
            if weight:
                self._set_goal_weight(t, round(weight + delta, 1))

    def _set_goal_weight(self, t: Turn, kg: float) -> None:
        height = t.profile.get("height_cm") or t.entities.get("height_cm")
        if height and kg / (height / 100) ** 2 < 18.5:
            t.notes.append(f"A goal weight of {kg:g} kg would put you in the underweight range for your height, so I did not save it.")
            return
        current, goal = t.profile.get("weight_kg"), t.profile.get("goal")
        if current and ((goal == "lose" and kg > current) or (goal == "gain" and kg < current)):
            t.notes.append("That goal weight points the opposite way to your goal, so check it.")
        if t.profile.get("goal_weight_kg") != kg:
            t.profile["goal_weight_kg"] = kg
            t.changed.append(self._describe("goal_weight_kg", kg, t.profile))

    def _dislikes(self, text: str) -> list[Food]:
        found: list[Food] = []
        for m in AVOID.finditer(text.lower()):
            phrase = m.group(1).strip()
            if not phrase or set(phrase.split()) & NOT_FOODS:
                continue
            hit = self.foods.match(phrase)
            if hit and hit[1] >= 0.95 and hit[0] not in found:
                found.append(hit[0])
        return found

    def _remember_dislikes(self, t: Turn) -> None:
        avoid = set(t.profile.get("avoid", []))
        for food in self._dislikes(t.text):
            if food.id not in avoid:
                avoid.add(food.id)
                t.changed.append(f"skipping {food_label(food.name)}")
        if avoid:
            t.profile["avoid"] = sorted(avoid)[:20]

    # ------------------------------------------------------------------ helpers
    def _targets(self, t: Turn) -> Optional[Targets]:
        if missing_fields(t.profile) or t.profile["age"] < 18:
            return None
        return compute_targets(t.profile)

    def _starter_chips(self, t: Turn) -> list[str]:
        if is_complete(t.profile):
            return ["Plan my meals", "Workout plan", "What's left today", "I had 2 eggs for breakfast"]
        return ["Calculate my calories", "Plan my meals", "Workout plan", "I had 2 eggs for breakfast"]

    def _ask_profile(self, t: Turn, then: str) -> None:
        missing = missing_fields(t.profile)
        names = [FIELD_NAMES[m] for m in missing]
        if len(missing) == len(CORE):
            t.say("To do that I need your age, sex, height and weight. Send them together, for example: "
                  "26, male, 178 cm, 78 kg, want to lose fat.")
        else:
            t.say(f"Thanks. I still need your {join_words(names)}.")
        t.pending = {"type": "need_profile", "then": then, "missing": missing}

    def _adult_gate(self, t: Turn) -> bool:
        age = t.profile.get("age")
        if age is not None and age < 18:
            t.say("My calorie and macro formulas are built for adults, 18 and over. For anyone younger, a doctor or "
                  "a registered dietitian is the right person to set targets. I can still help with a beginner "
                  "workout or with food logging.")
            t.chips = ["Workout plan", "I had 2 eggs for breakfast"]
            return False
        return True

    def _assumptions(self, t: Turn) -> list[str]:
        out = []
        if "goal" not in t.profile:
            out.append("No goal is set, so I used maintenance. Say 'I want to lose fat' or 'build muscle' to change it.")
        if "activity" not in t.profile and "days" not in t.profile:
            out.append("I assumed you are lightly active. Tell me how often you train to refine it.")
        return out

    def _restrictions(self, t: Turn) -> str:
        parts = []
        if t.profile.get("diet"):
            parts.append(DIET_TEXT[t.profile["diet"]])
        if t.profile.get("allergens"):
            parts.append("no " + join_words(ALLERGEN_TEXT.get(a, a) for a in t.profile["allergens"]))
        for fid in t.profile.get("avoid", []):
            if fid in self.foods.by_id:
                parts.append("no " + food_label(self.foods.by_id[fid].name))
        return ", ".join(parts)

    def _show_targets(self, t: Turn) -> None:
        tg = self._targets(t)
        if tg is None:
            return
        t.say(f"Your estimated maintenance is {n(tg.tdee)} kcal a day.")
        if tg.goal == "lose":
            t.say(f"For fat loss, aim for {n(tg.calories)} kcal, which is about {abs(tg.weekly_change_kg):.1f} kg a week.")
        elif tg.goal == "gain":
            t.say(f"For muscle gain, aim for {n(tg.calories)} kcal, which is about {tg.weekly_change_kg:.1f} kg a week.")
        else:
            t.say(f"To stay where you are, aim for {n(tg.calories)} kcal.")
        for note in self._assumptions(t):
            t.say(note)
        block = tg.to_dict()
        block["type"] = "targets"
        t.blocks.append(block)
        t.chips = ["Plan my meals", "Workout plan", "I had 2 eggs for breakfast"]

    def _today_totals(self, t: Turn, extra: Optional[list[dict]] = None) -> dict:
        totals = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for entry in [*t.state.log_today, *(extra or [])]:
            for k in totals:
                totals[k] += entry.get(k, 0.0)
        return totals

    def _pool(self, t: Turn, slot: Optional[str] = None) -> list[Food]:
        diet = t.entities.get("diet") or t.profile.get("diet")
        allergens = set(t.profile.get("allergens", [])) | set(t.entities.get("allergens") or [])
        return self.foods.allowed(diet, allergens, t.profile.get("avoid", []), slot=slot)

    # ----------------------------------------------------------------- handlers
    def _on_greet(self, t: Turn) -> None:
        if is_complete(t.profile):
            t.say("Hi. Want to plan today's meals, log what you ate, or get a workout?")
        else:
            t.say("Hi. Tell me your age, sex, height, weight and goal and I'll work out your targets.")
        t.chips = self._starter_chips(t)

    def _on_thanks(self, t: Turn) -> None:
        t.say("Anytime. What would you like to do next?")
        t.chips = self._starter_chips(t)

    def _on_bye(self, t: Turn) -> None:
        t.say("See you. Your profile and logs stay saved in this browser.")

    def _on_help(self, t: Turn) -> None:
        t.say("I can:",
              "Work out your calories and macros from your details",
              "Build a full day of meals that fits your targets",
              "Make a weekly workout plan",
              "Log what you eat from plain text",
              "Look up foods and exercises",
              "Track your weight and estimate your real maintenance calories")
        t.chips = self._starter_chips(t)

    def _on_about(self, t: Turn) -> None:
        t.say("I'm Ironplate, a nutrition and training assistant built by Abhishek Grover, AI ML Engineer.",
              "Everything runs on my own models: an intent classifier, a food search index, a meal-plan optimiser "
              "and a weight-trend model. No outside chat service is involved.")
        t.chips = self._starter_chips(t)

    def _on_fallback(self, t: Turn) -> None:
        t.say("I can help with food and training: targets, meal plans, workouts, food logging and progress.",
              "Try 'plan my meals' or 'I had 2 eggs for breakfast'.")
        t.chips = self._starter_chips(t)

    def _on_set_profile(self, t: Turn, then: Optional[str] = None) -> None:
        invalid = t.entities.get("invalid", [])
        if invalid:
            t.say("I couldn't use that " + join_words(FIELD_NAMES.get(i, i) for i in invalid)
                  + ". Please check the value.")
        recognised = any(k in t.entities for k in PROFILE_KEYS) or bool(t.changed)
        if not recognised and not invalid:
            if is_complete(t.profile):
                t.say("Tell me what to change, for example 'I'm vegetarian' or 'I weigh 78 kg'.")
                t.chips = self._starter_chips(t)
            else:
                self._ask_profile(t, then or "calc_targets")
            return
        if t.changed:
            t.say("Saved: " + ", ".join(t.changed) + ".")
        for note in t.notes:
            t.say(note)
        if then in {"meal_plan", "workout_plan", "calc_targets"} and (is_complete(t.profile) or then == "workout_plan"):
            self._handlers[then](t)
            return
        if is_complete(t.profile):
            if any(k in t.entities for k in (*CORE, "goal", "activity", "days", "pace", "goal_weight_kg")):
                if self._adult_gate(t):
                    self._show_targets(t)
            else:
                t.chips = self._starter_chips(t)
        elif any(k in t.entities for k in CORE):
            self._ask_profile(t, then or "calc_targets")
        else:
            t.chips = self._starter_chips(t)

    def _on_calc_targets(self, t: Turn) -> None:
        if not self._adult_gate(t):
            return
        if self._targets(t) is None:
            self._ask_profile(t, "calc_targets")
            return
        self._show_targets(t)

    def _on_meal_plan(self, t: Turn) -> None:
        if not self._adult_gate(t):
            return
        tg = self._targets(t)
        if tg is None:
            self._ask_profile(t, "meal_plan")
            return
        seed = t.plan_seed
        t.plan_seed += 1
        plan = plan_day(self.foods, tg.to_dict(), diet=t.profile.get("diet"),
                        allergens=t.profile.get("allergens", []), avoid=t.profile.get("avoid", []),
                        meals=t.profile.get("meals", 4), seed=seed)
        if plan is None:
            t.say("I couldn't build a day that satisfies those restrictions. Try removing one, for example "
                  "'clear restrictions', or ask for more meals.")
            t.chips = ["Clear restrictions", "Plan my meals"]
            return
        err, protein_err = plan["kcal_error_pct"], plan["protein_error_pct"]
        t.say(f"Here is a day built for {n(tg.calories)} kcal and {tg.protein_g} g protein.")
        limits = self._restrictions(t)
        if limits:
            t.say(f"Kept to: {limits}.")
        if protein_err < -10:
            short = tg.protein_g - plan["totals"]["protein"]
            t.say(f"Calories are on target, but protein comes out about {round(short)} g short because your "
                  f"restrictions leave few high-protein foods. A supplement or fewer exclusions would close the gap.")
        elif abs(err) < 0.5 and abs(protein_err) < 1:
            t.say("Calories and protein match your targets.")
        else:
            t.say(f"Calories land within {max(abs(err), 0.1):.1f}% of target and protein within {max(abs(protein_err), 1):.0f}%.")
        for note in self._assumptions(t):
            t.say(note)
        plan["type"] = "meal_plan"
        t.blocks.append(plan)
        t.chips = ["Another plan", "Workout plan", "I had 2 eggs for breakfast"]

    def _on_workout_plan(self, t: Turn) -> None:
        p = t.profile
        assumed: list[str] = []
        days = p.get("days")
        if not days:
            days = 3
            assumed.append("3 days a week")
        equipment = p.get("equipment")
        if not equipment:
            equipment = "gym"
            assumed.append("gym access")
        level = p.get("experience")
        if not level:
            level = "beginner"
            assumed.append("beginner level")
        seed = t.plan_seed
        t.plan_seed += 1
        plan = build_plan(self.exercises, days, equipment, level, p.get("goal"), p.get("focus"), seed)
        t.say(f"Here is a {plan['title']} plan ({plan['summary']}).")
        if assumed:
            t.say("I assumed " + join_words(assumed) + ". Say something like '4 days at home with dumbbells' to change it.")
        plan["type"] = "workout"
        t.blocks.append(plan)
        t.chips = ["Another workout plan", "Plan my meals", "How to do a squat"]

    def _on_log_food(self, t: Turn) -> None:
        parsed = self.foods.parse_meal(t.text, t.state.hour)
        if not parsed.items:
            if parsed.unmatched:
                names = ", ".join(f"'{u}'" for u in parsed.unmatched[:3])
                t.say(f"I couldn't find {names} in my food list. Try a simpler name, for example '2 rotis and dal'.")
            else:
                t.say("Tell me what you ate, for example: 2 eggs and a banana for breakfast.")
            t.chips = ["I had 2 eggs for breakfast", "Calories in banana"]
            return
        meal = parsed.meal or "snack"
        entries = []
        for item in parsed.items:
            s = item.food.summary(item.grams, item.portion)
            entries.append({"id": uuid.uuid4().hex[:10], "meal": meal, "name": s["name"], "grams": s["grams"],
                            "kcal": s["kcal"], "protein": s["protein"], "carbs": s["carbs"], "fat": s["fat"]})
        added = {k: sum(e[k] for e in entries) for k in ("kcal", "protein", "carbs", "fat")}
        t.actions.append({"type": "log_food", "date": t.state.today, "entries": entries})
        t.blocks.append({"type": "foods", "meal": meal, "items": [i.summary() for i in parsed.items],
                         "totals": {k: round(v, 1) if k != "kcal" else round(v) for k, v in added.items()},
                         "entry_ids": [e["id"] for e in entries]})
        t.say(f"Logged {meal}: {n(added['kcal'])} kcal, {added['protein']:.0f} g protein.")
        inexact = [i for i in parsed.items if i.score < 0.95]
        if inexact:
            t.say("Closest match used: " + "; ".join(f"'{i.query}' as {food_label(i.food.name)}" for i in inexact) + ".")
        if parsed.unmatched:
            t.say("I couldn't find: " + ", ".join(f"'{u}'" for u in parsed.unmatched[:3]) + ".")
        tg = self._targets(t)
        if tg:
            totals = self._today_totals(t, entries)
            left = tg.calories - totals["kcal"]
            t.say(f"Today so far: {n(totals['kcal'])} of {n(tg.calories)} kcal, "
                  + (f"{n(left)} left." if left >= 0 else f"{n(-left)} over.")
                  + f" Protein {totals['protein']:.0f} of {tg.protein_g} g.")
        t.chips = ["What's left today", "Plan my meals", "I weigh 77 kg today"]

    def _food_candidates(self, text: str):
        options = [self.foods.strip_question_words(text), self.foods.strip_question_words(text, keep_nutrients=True), text]
        best = None
        for option in options:
            if not option.strip():
                continue
            parsed = self.foods.parse_meal(option)
            if parsed.items:
                score = min(i.score for i in parsed.items) + 0.01 * len(parsed.items)
                if best is None or score > best[0]:
                    best = (score, parsed)
        return best[1] if best else None

    def _on_food_info(self, t: Turn) -> None:
        parsed = self._food_candidates(t.text)
        if parsed is None or not parsed.items:
            t.say("I couldn't find that food. Try a simpler name, for example 'calories in paneer'.")
            t.chips = ["Calories in banana", "Protein in paneer"]
            return
        low = t.text.lower()
        focus = next((k for k, rx in (("protein", r"protein"), ("carbs", r"carb"), ("fat", r"\bfat"), ("fiber", r"fib")) if re.search(rx, low)), None)
        quantity = bool(re.search(r"\d|\b(?:two|three|four|half|couple)\b", low))
        if len(parsed.items) == 1 and not quantity:
            food = parsed.items[0].food
            serving = food.summary(food.unit_g)
            per100 = food.summary(100.0)
            t.blocks.append({"type": "food_info", "name": food.label, "per100": per100, "serving": serving,
                             "query": parsed.items[0].query, "exact": parsed.items[0].score >= 0.95})
            if focus:
                label = {"protein": "protein", "carbs": "carbs", "fat": "fat", "fiber": "fibre"}[focus]
                key = "fiber" if focus == "fiber" else focus
                t.say(f"{food.label} has {per100[key]:g} g {label} per 100 g"
                      + ("." if abs(serving["grams"] - 100) < 1 else f", and {serving[key]:g} g in {serving['portion']}."))
            else:
                t.say(f"{food.label} per 100 g: {per100['kcal']} kcal, {per100['protein']:g} g protein, "
                      f"{per100['carbs']:g} g carbs, {per100['fat']:g} g fat.")
                if abs(serving["grams"] - 100) >= 1:
                    t.say(f"One serving ({serving['portion']}): {serving['kcal']} kcal.")
            t.chips = ["Log " + log_phrase(serving), "Plan my meals"]
        else:
            items = [i.summary() for i in parsed.items]
            totals = {k: round(sum(i[k] for i in items), 1) for k in ("protein", "carbs", "fat")}
            totals["kcal"] = round(sum(i["kcal"] for i in items))
            t.blocks.append({"type": "foods", "meal": None, "items": items, "totals": totals, "entry_ids": []})
            t.say(f"That comes to {n(totals['kcal'])} kcal, {totals['protein']:.0f} g protein, "
                  f"{totals['carbs']:.0f} g carbs and {totals['fat']:.0f} g fat.")
            t.chips = ["Log " + ", ".join(log_phrase(i) for i in items[:3]), "Plan my meals"]

    def _on_food_suggest(self, t: Turn) -> None:
        low = t.text.lower()
        swap = re.search(r"(?:instead of|alternatives? (?:to|for)|substitutes? (?:for)?|swap(?: for)?|replace|replacement for|in place of)\s+(.+)$", low)
        if swap:
            hit = self.foods.match(re.sub(r"\b(?:with|by)\b.*$", "", swap.group(1)).strip(" ?.!"))
            if hit:
                food = hit[0]
                near = self.foods.similar(food, self._pool(t), 5)
                if near:
                    rows = []
                    for other, _ in near:
                        s = other.summary(other.unit_g)
                        dk, dp, df = other.kcal - food.kcal, other.protein - food.protein, other.fiber - food.fiber
                        s["note"] = f"{dk:+.0f} kcal, {dp:+.1f} g protein, {df:+.1f} g fibre per 100 g"
                        rows.append(s)
                    t.say(f"Closest swaps for {food_label(food.name)}, compared per 100 g:")
                    t.blocks.append({"type": "foods_list", "title": "Swaps", "items": rows})
                    t.chips = ["Plan my meals", "High protein snacks"]
                    return
        metric = "healthy"
        for name, rx in (("lowcarb", r"low[- ]carb|keto"), ("lowfat", r"low[- ]fat"), ("healthyfat", r"healthy fat|good fat|omega"),
                         ("lowcal", r"low[- ]cal|light|weight loss|fat loss|lose weight"), ("fiber", r"fib(?:er|re)|filling|full"),
                         ("protein", r"protein|muscle")):
            if re.search(rx, low):
                metric = name
                break
        slot = next((m for m in MEALS if m in low), None)
        pool = self._pool(t, slot)
        if slot is None:
            pool = [f for f in pool if f.role != "fat" or metric == "healthyfat"]
        foods = self.foods.rank(metric, pool, 6)
        if not foods:
            t.say("I couldn't find anything that fits those filters. Try fewer restrictions.")
            return
        rows = []
        for f in foods:
            s = f.summary(f.unit_g)
            s["note"] = {"protein": f"{f.protein / max(f.kcal, 1) * 100:.0f} g protein per 100 kcal",
                         "fiber": f"{f.fiber / max(f.kcal, 1) * 100:.1f} g fibre per 100 kcal",
                         "lowcal": f"{s['kcal']} kcal a serving",
                         "lowcarb": f"{s['carbs']:g} g carbs a serving",
                         "lowfat": f"{s['fat']:g} g fat a serving",
                         "healthyfat": f"{s['fat']:g} g fat a serving",
                         "healthy": f"{s['protein']:g} g protein, {s['fiber']:g} g fibre"}[metric]
            rows.append(s)
        title = {"protein": "High-protein", "fiber": "High-fibre", "lowcal": "Low-calorie", "lowcarb": "Low-carb",
                 "lowfat": "Low-fat", "healthyfat": "Healthy-fat", "healthy": "Nutrient-dense"}[metric]
        title += f" {slot} ideas" if slot else " foods"
        limits = self._restrictions(t)
        t.say(title + (f", kept to {limits}:" if limits else ":"))
        t.blocks.append({"type": "foods_list", "title": title, "items": rows})
        t.chips = ["Plan my meals", "Calories in " + food_label(foods[0].name)]

    def _weights(self, t: Turn) -> list[tuple]:
        series = list(t.state.weights)
        for action in t.actions:
            if action["type"] == "log_weight":
                series.append((action["date"], action["kg"]))
        return clean_series(series)

    def _on_log_weight(self, t: Turn) -> None:
        kg = t.entities.get("weight_kg")
        if kg is None:
            t.say("What is your weight today? For example: I weigh 76.5 kg.")
            return
        shown = self._describe("weight_kg", kg, t.profile)
        t.say(f"Logged {shown} for today.")
        series = self._weights(t)
        tg = self._targets(t)
        offset = (tg.calories - tg.tdee) if tg else None
        a = analyze(series, {}, t.profile.get("goal"), t.profile.get("goal_weight_kg"), offset)
        if a["status"] == "ok":
            t.say(f"Trend weight is {a['trend_kg']:g} kg, changing about {a['weekly_rate_kg']:+.2f} kg a week.")
        else:
            t.say("Log a few more days and I'll show your trend.")
        t.chips = ["How am I doing", "What's left today", "Plan my meals"]

    def _on_progress(self, t: Turn) -> None:
        series = self._weights(t)
        tg = self._targets(t)
        today = parse_day(t.state.today)
        intake = {parse_day(d): k for d, k in t.state.intake if parse_day(d) and parse_day(d) != today}
        a = analyze(series, intake, t.profile.get("goal"), t.profile.get("goal_weight_kg"),
                    (tg.calories - tg.tdee) if tg else None)
        if a["status"] != "ok":
            t.say(f"I need at least three weigh-ins spread over a week or more, and you have {a['n']}.",
                  "Log your weight with something like 'I weigh 77.4 kg'.")
            t.chips = ["I weigh 77 kg today", "Plan my meals"]
            return
        t.say(f"Trend weight is {a['trend_kg']:g} kg, changing about {a['weekly_rate_kg']:+.2f} kg a week "
              f"(likely between {a['rate_low']:+.2f} and {a['rate_high']:+.2f}).")
        if a["eta_weeks"]:
            t.say(f"At this pace you would reach {a['goal_weight_kg']:g} kg in about {a['eta_weeks']} weeks.")
        advice = {
            "on_track": "That is a healthy pace for your goal.",
            "too_fast": "That is faster than about 1% of body weight a week for fat loss, or 0.5% for gain. Faster "
                        "change tends to cost muscle or add fat, so consider adding 100 to 200 kcal.",
            "plateau": "Your weight has been flat for two weeks. If your logging is accurate, lower calories by 100 to 150 "
                       "kcal or raise daily steps, for a gain goal raise calories by about the same.",
            "wrong_direction": "Weight is moving the opposite way to your goal. Check portion sizes and weekend intake before "
                               "changing your targets.",
            "steady": "Your weight is steady.", "drifting": "Your weight is drifting more than 0.5 kg a week.",
        }
        for flag in a["flags"]:
            if flag in advice:
                t.say(advice[flag])
        ad = a["adaptive"]
        if ad:
            t.say(f"From your logs, your real maintenance is about {n(ad['tdee_est'])} kcal"
                  + (f" (the formula says {n(tg.tdee)})" if tg else "") + f". Confidence: {ad['confidence']}.")
            if ad["suggested_calories"]:
                t.say(f"Suggested daily target: {n(ad['suggested_calories'])} kcal.")
        else:
            t.say("Log food for about two weeks alongside your weight and I'll estimate your real maintenance calories.")
        t.blocks.append({"type": "trend", **{k: a[k] for k in ("points", "trend_kg", "weekly_rate_kg", "goal_weight_kg", "eta_weeks")},
                         "adaptive": ad})
        t.chips = ["I weigh 77 kg today", "What's left today", "Plan my meals"]

    def _on_today_summary(self, t: Turn) -> None:
        entries = t.state.log_today
        if not entries:
            t.say("Nothing is logged today yet. Tell me what you ate, for example: 2 eggs and a banana for breakfast.")
            t.chips = ["I had 2 eggs for breakfast", "Plan my meals"]
            return
        totals = self._today_totals(t)
        meals = []
        for meal in MEALS:
            rows = [e for e in entries if e["meal"] == meal]
            if rows:
                meals.append({"name": meal.capitalize(), "kcal": round(sum(e["kcal"] for e in rows)),
                              "items": [{"name": e["name"], "grams": round(e["grams"]), "kcal": round(e["kcal"])} for e in rows]})
        tg = self._targets(t)
        t.blocks.append({"type": "today", "meals": meals, "totals": {k: round(v, 1) if k != "kcal" else round(v) for k, v in totals.items()},
                         "targets": ({"calories": tg.calories, "protein_g": tg.protein_g, "carbs_g": tg.carbs_g, "fat_g": tg.fat_g} if tg else None)})
        if not tg:
            t.say(f"So far today: {n(totals['kcal'])} kcal and {totals['protein']:.0f} g protein.",
                  "Add your details and I'll show what is left.")
            t.chips = ["Calculate my calories"]
            return
        left = tg.calories - totals["kcal"]
        gap = tg.protein_g - totals["protein"]
        t.say(f"{n(totals['kcal'])} of {n(tg.calories)} kcal so far, " + (f"{n(left)} left." if left >= 0 else f"{n(-left)} over."),
              f"Protein {totals['protein']:.0f} of {tg.protein_g} g.")
        if gap > 15 and left > 100:
            def fits(f: Food) -> bool:
                s = f.nutrients(f.unit_g)
                return s["kcal"] <= max(left, 100) and s["protein"] >= 8

            picks = [f for f in self.foods.rank("protein", self._pool(t, "snack"), 14) if fits(f)][:3]
            if len(picks) < 3:
                picks += [f for f in self.foods.rank("protein", self._pool(t), 20) if fits(f) and f not in picks][:3 - len(picks)]
            if picks:
                t.say("To close the protein gap: " + "; ".join(
                    f"{food_label(f.name)} ({f.portion_label(f.unit_g)}, {f.nutrients(f.unit_g)['protein']:.0f} g protein)" for f in picks) + ".")
        t.chips = ["High protein snacks", "Plan my meals", "I weigh 77 kg today"]

    def _on_exercise_info(self, t: Turn) -> None:
        low = t.text.lower()
        words = re.sub(r"\b(?:how|to|do|does|a|an|the|form|proper|exercises?|for|best|show|me|what|is|are|muscles?|work|works|"
                       r"alternatives?|alternative|tips|explain|teach|list|some|which|build|at|home|with|day|i|perform|easier|"
                       r"good|top|my|workout|routine|body)\b", " ", low)
        stripped = re.sub(r"\s+", " ", words).strip()
        equipment = t.entities.get("equipment") or t.profile.get("equipment") or "gym"
        muscles = self.exercises.muscles_in(low)
        hit = self.exercises.search(stripped) if stripped else None
        only_muscles = not stripped or all(self.exercises.muscles_in(w) for w in stripped.split())
        if hit and not only_muscles:
            ex = hit[0]
            alts = self.exercises.alternatives(ex, equipment)
            t.say(f"{ex.label} mainly works your {ex.primary}"
                  + (f" and also your {join_words(ex.secondary)}." if ex.secondary else "."), ex.cue)
            t.blocks.append({"type": "exercise", "name": ex.label, "primary": ex.primary, "secondary": list(ex.secondary),
                             "cue": ex.cue, "equipment": ex.equipment, "alternatives": [a.label for a in alts]})
            t.chips = [f"{alts[0].label} form" if alts else "Workout plan", "Workout plan"]
            return
        if muscles:
            groups = []
            for m in muscles[:3]:
                items = self.exercises.for_muscle(m, equipment, 5)
                if items:
                    groups.append((m, items))
            if groups:
                label = join_words(m for m, _ in groups)
                t.say(f"Good exercises for {label} ({EQUIPMENT_LABEL[equipment]}):")
                rows = [{"name": e.label, "cue": e.cue, "note": m} for m, items in groups for e in items]
                t.blocks.append({"type": "exercise_list", "title": f"{label.capitalize()} exercises", "items": rows})
                t.chips = ["Workout plan", f"How to do {groups[0][1][0].name}"]
                return
        t.say("I couldn't find that exercise. Try a name like 'deadlift' or a muscle like 'chest exercises'.")
        t.chips = ["Chest exercises", "How to do a squat"]

    def _on_reset(self, t: Turn) -> None:
        t.say("This clears your profile, food log and weights from this browser. Reply 'yes' to confirm or 'no' to keep them.")
        t.pending = {"type": "confirm_reset"}
        t.chips = ["Yes", "No"]
