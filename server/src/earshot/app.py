"""FastAPI app: the device WebSocket (/ws), a health check, and the web client as static files."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import agent
from .budget import Budget
from .config import DATA_DIR, WEB_DIR, MissingKey, load_config
from .research import Researcher
from .session import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("earshot")

CFG = load_config()
BUDGET = Budget(CFG["budget"]["daily_usd"], DATA_DIR / "spend.json")
OPEN: Counter[str] = Counter()  # live sessions per client IP
# Shared: research jobs outlive the connection that started them, and (kept on disk) the server process too.
RESEARCHER = Researcher(CFG, BUDGET, store=DATA_DIR / "research")
BRAIN_CHECK_S = 300


async def watch_brain() -> None:
    """Keeps turns off a brain the API won't serve (no wasted 404 per turn) and back on it once it answers."""
    brain = CFG["brain"]
    while True:
        try:
            ok = await agent.check_model(CFG, brain)
        except MissingKey:
            ok = False
        if ok is False:
            agent.UNAVAILABLE[brain] = time.monotonic() + BRAIN_CHECK_S * 2
        elif ok:
            agent.UNAVAILABLE.pop(brain, None)
        log.info("brain %s check: %s", brain, {True: "live", False: "refused, using fallback", None: "unsure"}[ok])
        await asyncio.sleep(BRAIN_CHECK_S)


@asynccontextmanager
async def lifespan(_: FastAPI):
    watcher = asyncio.create_task(watch_brain())
    if n := RESEARCHER.resume():
        log.info("resumed %d research job(s) from before the restart", n)
    yield
    RESEARCHER.shutting_down = True  # jobs in flight keep running at Meta; the next start collects them
    watcher.cancel()


app = FastAPI(title="Earshot", lifespan=lifespan)


@app.middleware("http")
async def revalidate_client(request, call_next):
    """Browsers otherwise keep a stale page after a redeploy; `no-cache` still allows cheap 304s."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/healthz")
async def healthz() -> dict:
    live = agent.UNAVAILABLE.get(CFG["brain"], 0) <= time.monotonic()
    return {"ok": True, "brain": CFG["brain"] if live else CFG.get("fallback"), "brain_live": live,
            "spent_today_usd": round(BUDGET.spent, 4), "budget_usd": BUDGET.daily_usd,
            "open_sessions": sum(OPEN.values()), "research_running": len(RESEARCHER.tasks)}


def client_ip(ws: WebSocket) -> str:
    fwd = ws.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (ws.client.host if ws.client else "?")


@app.websocket("/ws")
async def device_socket(ws: WebSocket) -> None:
    ip = client_ip(ws)
    if OPEN[ip] >= CFG["budget"]["sessions_per_ip"]:
        await ws.close(code=1013, reason="too many sessions from this address")
        return
    await ws.accept()
    OPEN[ip] += 1
    outbox: asyncio.Queue[dict] = asyncio.Queue()

    async def send(msg: dict) -> None:
        await outbox.put(msg)

    async def writer() -> None:  # one writer, so tasks never interleave frames on the socket
        while True:
            await ws.send_text(json.dumps(await outbox.get()))

    writer_task = asyncio.create_task(writer())
    session = Session(cfg=CFG, budget=BUDGET, send=send, researcher=RESEARCHER)
    clean = False  # a clean close forgets the conversation; a dropped one keeps it for the reconnect
    try:
        await session.start(await ws.receive_json())
        deadline = time.monotonic() + CFG["budget"]["web_session_minutes"] * 60 if session.client == "web" else None
        log.info("session open: %s from %s", session.client, ip)
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                clean = msg.get("code") in (1000, 1001)  # the page closed or changed language
                break
            if msg.get("bytes") is not None:
                session.handle_audio(msg["bytes"])
            elif msg.get("text") is not None:
                await session.handle_text(json.loads(msg["text"]))
            if deadline and time.monotonic() > deadline:
                await session.expire()
                await asyncio.sleep(0.5)  # let the writer deliver the goodbye
                clean = True
                break
    except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError) as e:
        log.info("session ended: %r", e)
    finally:
        await session.close(clean=clean)
        writer_task.cancel()
        OPEN[ip] -= 1
        try:
            await ws.close()
        except RuntimeError:
            pass


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
