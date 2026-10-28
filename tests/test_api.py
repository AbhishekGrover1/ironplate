import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(scope="module")
def api():
    with TestClient(main.app) as c:
        yield c


def chat(api, message, **state):
    return api.post("/api/chat", json={"message": message, "state": state})


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok" and body["intents"] == 18 and body["foods"] > 100


def test_chat_round_trip_with_browser_state(api):
    r = chat(api, "I'm 26, male, 178 cm, 78 kg, want to lose fat", today="2026-10-16", hour=9)
    body = r.json()
    assert r.status_code == 200 and body["targets"]["calories"] > 1500 and body["profile"]["age"] == 26
    plan = chat(api, "plan my meals", profile=body["profile"], plan_seed=body["plan_seed"]).json()
    assert plan["blocks"][0]["type"] == "meal_plan" and plan["plan_seed"] == 1


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "x" * 601}, {}, {"message": 5}])
def test_invalid_requests_are_rejected(api, payload):
    assert api.post("/api/chat", json=payload).status_code == 422


def test_garbage_state_cannot_break_the_server(api):
    r = chat(api, "hello", profile="oops", weights=5, log_today=[1, 2], pending={"type": "evil"}, hour=99, today="nope")
    assert r.status_code == 200 and r.json()["intent"] == "greet"


def test_rate_limit(api, monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter(2, 60))
    codes = [chat(api, "hi").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_food_search_endpoint(api):
    hits = api.get("/api/foods", params={"q": "banan"}).json()["results"]
    assert hits[0]["name"] == "Banana" and hits[0]["score"] > 0.5


def test_model_metrics_are_published(api):
    cv = api.get("/api/model").json()["cross_validation"]
    assert 0.5 < cv["accuracy"] <= 1 and "per_intent" in cv


def test_front_end_is_served_with_security_headers(api):
    page = api.get("/")
    assert page.status_code == 200 and "Ironplate" in page.text
    assert page.headers["x-frame-options"] == "DENY" and "default-src 'self'" in page.headers["content-security-policy"]
    assert api.get("/script.js").status_code == 200 and api.get("/style.css").status_code == 200
