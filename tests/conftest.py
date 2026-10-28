"""Shared fixtures: a simulated browser client that keeps state the way the web page does."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dialogue import Assistant  # noqa: E402
from app.schemas import ClientState  # noqa: E402


class Client:
    """Mimics the browser: profile, pending state, food log and weights live here."""

    def __init__(self, assistant: Assistant, today: str = "2026-10-16", hour: int = 9):
        self.assistant = assistant
        self.profile: dict = {}
        self.pending = None
        self.plan_seed = 0
        self.today = today
        self.hour = hour
        self.log: dict[str, list[dict]] = {}
        self.weights: dict[str, float] = {}

    def say(self, message: str) -> dict:
        intake = [(d, sum(e["kcal"] for e in rows)) for d, rows in self.log.items() if d != self.today]
        state = ClientState.from_raw({
            "profile": self.profile, "pending": self.pending, "plan_seed": self.plan_seed,
            "today": self.today, "hour": self.hour, "log_today": self.log.get(self.today, []),
            "weights": [(d, kg) for d, kg in self.weights.items()], "intake": intake,
        })
        out = self.assistant.respond(message, state)
        self.profile, self.pending, self.plan_seed = out["profile"], out["pending"], out["plan_seed"]
        for action in out["actions"]:
            if action["type"] == "log_food":
                self.log.setdefault(action["date"], []).extend(action["entries"])
            elif action["type"] == "log_weight":
                self.weights[action["date"]] = action["kg"]
            elif action["type"] == "reset_all":
                self.profile, self.log, self.weights = {}, {}, {}
        return out

    def days_ago(self, n: int) -> str:
        return (date.fromisoformat(self.today) - timedelta(days=n)).isoformat()


@pytest.fixture(scope="session")
def assistant() -> Assistant:
    return Assistant.load()


@pytest.fixture()
def client(assistant) -> Client:
    return Client(assistant)
