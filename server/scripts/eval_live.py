"""Does the live demo hold up? Runs the judge's prompts against a running server, several times each.

    uv run python scripts/eval_live.py --url wss://earshot-yuansshi.fly.dev/ws --trials 3

Spoken prompts are synthesized with macOS `say` and streamed in real time, like a phone would. Device tools get
canned answers (a Seattle location; reminders always save). Each trial is its own session and device id. At most
3 run at once, the server's per-address limit. Prints a table and writes `results/live-eval.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe import synth_pcm  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT.parent / "web" / "samples" / "platform.jpg"
LOCATION = {"latitude": 47.6062, "longitude": -122.3321, "accuracy_m": 30, "place": "Downtown, Seattle"}


class Trial:
    """One session: records every server event with its time, answers device tools, and answers yes when asked."""

    def __init__(self, ws, t0: float):
        self.ws, self.t0, self.events = ws, t0, []

    async def read(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            self.events.append((time.perf_counter(), msg))
            if msg.get("type") == "tool_call":
                result = LOCATION if msg["name"] == "get_location" else {"ok": True, "list": "Reminders"}
                await self.ws.send(json.dumps({"type": "tool_result", "id": msg["id"], "ok": True, "result": result}))
            elif msg.get("type") == "confirm":
                await self.ws.send(json.dumps({"type": "text", "text": "yes"}))

    async def wait(self, test, timeout: float, after: float = 0.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            for t, msg in self.events:
                if t >= after and test(msg):
                    return t, msg
            await asyncio.sleep(0.05)
        raise TimeoutError

    async def speak(self, pcm: bytes) -> float:
        for i in range(0, len(pcm), 640):
            await self.ws.send(pcm[i:i + 640])
            await asyncio.sleep(0.02)
        return time.perf_counter()

    async def text(self, text: str) -> float:
        await self.ws.send(json.dumps({"type": "text", "text": text}))
        return time.perf_counter()

    def turn(self, after: float, done_at: float) -> dict:
        """First word, total, cost and tools of the turn(s) that ended by `done_at`."""
        says = [t for t, m in self.events if after <= t <= done_at and m.get("type") == "say" and m.get("text")]
        dones = [m for t, m in self.events if after <= t <= done_at and m.get("type") == "turn_done"]
        words = " ".join(m["text"] for t, m in self.events if after <= t <= done_at and m.get("type") == "say")
        return {"first_s": round(says[0] - after, 2) if says else None, "total_s": round(done_at - after, 2),
                "cost": round(sum(d.get("cost_usd") or 0 for d in dones), 5),
                "tools": sorted({tool for d in dones for tool in d.get("tools") or []}), "said": words[:300]}


def done(msg: dict, background: bool = False) -> bool:
    return msg.get("type") == "turn_done" and bool(msg.get("background")) == background and not msg.get("heads_up")


async def wake_weather(tr: Trial, pcm: dict) -> dict:
    end = await tr.speak(pcm["weather"])
    t, _ = await tr.wait(done, 40, end)
    r = tr.turn(end, t)
    return {**r, "ok": "get_weather" in r["tools"] and bool(r["said"])}


async def look_sample(tr: Trial, pcm: dict) -> dict:
    image = "data:image/jpeg;base64," + base64.b64encode(SAMPLE.read_bytes()).decode()
    sent = time.perf_counter()
    await tr.ws.send(json.dumps({"type": "image", "data": image,
                                 "text": "Which platform is the 19:05? Show me where it is on the board."}))
    t, _ = await tr.wait(done, 40, sent)
    r = tr.turn(sent, t)
    platform = any(w in r["said"].lower() for w in ("platform three", "platform 3"))
    return {**r, "ok": platform and "find_in_photo" in r["tools"], "platform_right": platform}


async def remember_parking(tr: Trial, pcm: dict) -> dict:
    first = await tr.text("Remember I parked on level 3, spot 42.")
    await tr.wait(done, 30, first)
    ask = await tr.text("Where did I park?")
    t, _ = await tr.wait(done, 30, ask + 0.01)
    r = tr.turn(ask, t)
    said = r["said"].lower()
    return {**r, "ok": ("3" in said or "three" in said) and ("42" in said or "forty" in said)}


async def rewind(tr: Trial, pcm: dict) -> dict:
    await tr.speak(pcm["friend_address"])
    await tr.wait(lambda m: m.get("type") == "transcript" and m.get("final") and m.get("reason") == "no_wake", 15)
    end = await tr.speak(pcm["what_address"])
    t, _ = await tr.wait(done, 40, end)
    r = tr.turn(end, t)
    said = r["said"].lower()
    return {**r, "ok": "rewind" in r["tools"] and ("pine" in said or "1420" in said or "fourteen twenty" in said)}


async def heads_up(tr: Trial, pcm: dict) -> dict:
    end = await tr.speak(pcm["friend_dinner"])
    asked, _ = await tr.wait(lambda m: m.get("type") == "confirm", 20, end)
    t, _ = await tr.wait(done, 40, asked)
    r = tr.turn(asked, t)
    return {**r, "offer_s": round(asked - end, 2), "ok": "create_reminder" in r["tools"]}


async def wonder(tr: Trial, pcm: dict) -> dict:
    sent = await tr.text("Look into the best e-bikes under $1,500 and get back to me.")
    t, msg = await tr.wait(lambda m: done(m, background=True), 240, sent)
    return {"ok": any(m.get("type") == "report" for _, m in tr.events), "total_s": round(t - sent, 1),
            "cost": msg.get("cost_usd"), "searches": msg.get("searches"), "tools": ["research"], "first_s": None}


SCENARIOS = {"Wake word + weather (spoken)": wake_weather, "Look + Find it (sample photo)": look_sample,
             "Remember where I parked": remember_parking, "Rewind (spoken)": rewind,
             "Heads-up (spoken)": heads_up, "Muse Wonder e-bike research": wonder}


async def run_one(url: str, name: str, pcm: dict, gate: asyncio.Semaphore) -> dict:
    async with gate:
        for attempt in (1, 2, 3):  # a dropped session (the trial server restarts) or a full server: wait, retry
            if attempt > 1:
                await asyncio.sleep(4 * attempt)
            t0 = time.perf_counter()
            try:
                async with connect(url, open_timeout=30, max_size=None) as ws:
                    await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "wake",
                                              "device_id": f"eval-{uuid.uuid4().hex[:8]}", "tz": "America/Los_Angeles",
                                              "sample_rate": 16000, "tools": ["get_location", "create_reminder"],
                                              "heads_up": name.startswith("Heads-up")}))
                    tr = Trial(ws, t0)
                    reader = asyncio.create_task(tr.read())
                    await tr.wait(lambda m: m.get("type") == "ready", 30)
                    await asyncio.sleep(1.5)  # the ASR session opens
                    try:
                        result = await SCENARIOS[name](tr, pcm)
                    finally:
                        reader.cancel()
                    return {"scenario": name, "attempt": attempt, **result}
            except Exception as e:
                error = f"{type(e).__name__}: {e}"[:120]
        return {"scenario": name, "ok": False, "error": error, "first_s": None, "total_s": None, "cost": 0}


def p50(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.1f} s" if xs else "–"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://earshot-yuansshi.fly.dev/ws")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", help="run scenarios whose name contains this text")
    ap.add_argument("--out", default=str(ROOT / "results" / "live-eval.json"))
    args = ap.parse_args()
    pcm = {k: synth_pcm(v, 16000) for k, v in {
        "weather": "Hey Earshot, will it rain this afternoon?",
        "friend_address": "Meet me at 1420 Pine Street, unit 3, at seven tonight.",
        "what_address": "Hey Earshot, what was that address she just said?",
        "friend_dinner": "Don't forget, dinner's at my place on Friday at seven. It's 1420 Pine Street, unit 3."}.items()}
    names = [n for n in SCENARIOS if not args.only or args.only.lower() in n.lower()]
    gate = asyncio.Semaphore(3)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*(run_one(args.url, n, pcm, gate) for n in names for _ in range(args.trials)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": args.url, "rows": rows}, indent=1))
    print("| Prompt | Passed | First word (p50) | Done (p50) | Cost per run |")
    print("|---|---|---|---|---|")
    for n in names:
        rs = [r for r in rows if r["scenario"] == n]
        costs = [r["cost"] for r in rs if r.get("cost") is not None]
        extra = f" (offer after {p50([r.get('offer_s') for r in rs])})" if n.startswith("Heads-up") else ""
        print(f"| {n} | {sum(r['ok'] for r in rs)}/{len(rs)} | {p50([r['first_s'] for r in rs])}{extra} | "
              f"{p50([r['total_s'] for r in rs])} | ${statistics.mean(costs) if costs else 0:.4f} |")
    print(f"{len(rows)} trials in {time.perf_counter() - t0:.0f} s; failures:",
          [(r['scenario'], r.get('error') or r.get('said', '')[:80]) for r in rows if not r["ok"]])


if __name__ == "__main__":
    asyncio.run(main())
