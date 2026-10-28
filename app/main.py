"""FastAPI application: chat API, food search, health check and the static front end."""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import MODEL_DIR, STATIC_DIR
from .dialogue import Assistant
from .schemas import ChatRequest, ChatResponse, ClientState

log = logging.getLogger("ironplate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

RATE_LIMIT = 60
RATE_WINDOW = 60.0
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; base-uri 'self'; "
       "form-action 'self'; frame-ancestors 'none'")


class RateLimiter:
    """Sliding-window limiter kept in memory, per client address."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        if len(self._hits) > 5000:
            self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
        return True


limiter = RateLimiter(RATE_LIMIT, RATE_WINDOW)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    app.state.assistant = Assistant.load()
    log.info("Models and data loaded in %.1fs", time.perf_counter() - started)
    yield


app = FastAPI(title="Ironplate", version=__version__, lifespan=lifespan, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if not request.url.path.startswith(("/docs", "/openapi.json")):
        response.headers.setdefault("Content-Security-Policy", CSP)
    return response


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


@app.get("/health")
def health(request: Request) -> dict:
    a: Assistant = request.app.state.assistant
    return {"status": "ok", "version": __version__, "foods": len(a.foods.foods),
            "exercises": len(a.exercises.items), "intents": len(a.model.classes)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> dict:
    if not limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down for a moment.")
    state = ClientState.from_raw(body.state)
    try:
        return request.app.state.assistant.respond(body.message, state)
    except Exception:
        log.exception("chat request failed")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.get("/api/foods")
def search_foods(request: Request, q: str = Query(min_length=1, max_length=80), k: int = Query(5, ge=1, le=10)) -> dict:
    """Fuzzy food search over the character n-gram index."""
    hits = request.app.state.assistant.foods.search(q, k)
    return {"query": q, "results": [{**f.summary(100.0), "score": round(score, 3)} for f, score in hits]}


@app.get("/api/model")
def model_info() -> dict:
    """Training metrics written by `python train.py`."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics yet.")
    return json.loads(path.read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
