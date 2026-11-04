"""Cold start: stop the Fly machine, then time a judge's first visit, from the page to the first spoken word.

    uv run python scripts/eval_cold_start.py --app earshot-yuansshi --trials 3

Needs flyctl, logged in. Each trial stops every machine, waits until they're stopped, then:
1. loads the page (Fly starts the machine on this request),
2. opens the WebSocket and says hello, as the page's Start button does,
3. types a question and waits for the first spoken word, then the end of the turn.
Writes eval/results/cold-start-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]


def machines(app: str) -> list[dict]:
    out = subprocess.run(["fly", "machines", "list", "--app", app, "--json"], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def stop_all(app: str) -> None:
    for m in machines(app):
        if m["state"] == "started":
            subprocess.run(["fly", "machine", "stop", m["id"], "--app", app], capture_output=True, text=True)
    for _ in range(90):
        if all(m["state"] in ("stopped", "suspended") for m in machines(app)):
            return
        time.sleep(2)
    raise RuntimeError("the machines didn't stop")


async def visit(base: str) -> dict:
    t0 = time.perf_counter()
    urllib.request.urlopen(base + "/", timeout=90).read()
    page_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    async with connect(base.replace("https://", "wss://") + "/ws", open_timeout=60) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "ptt", "device_id": f"cold-{uuid.uuid4().hex[:8]}",
                                  "tz": "America/Los_Angeles", "sample_rate": 16000, "tools": []}))
        first = ask = None
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "ready":
                ready_s = time.perf_counter() - t1
                ask = time.perf_counter()
                await ws.send(json.dumps({"type": "text", "text": "In one sentence, what can you help me with?"}))
            elif msg["type"] == "say" and msg.get("text") and first is None:
                first = time.perf_counter() - ask
            elif msg["type"] == "turn_done":
                break
    return {"page_s": round(page_s, 2), "ready_s": round(ready_s, 2), "first_word_s": round(first, 2),
            "turn_s": round(time.perf_counter() - ask, 2), "page_to_first_word_s": round(time.perf_counter() - t0, 2)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="earshot-yuansshi")
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    base = f"https://{a.app}.fly.dev"
    rows = []
    for k in range(a.trials):
        stop_all(a.app)
        await asyncio.sleep(5)
        row = await visit(base)
        warm = await visit(base)  # the same visit again, now that the machine is up
        rows.append({"cold": row, "warm": warm})
        print(k + 1, "cold", row, "\n  warm", warm)
    med = {f"{kind}_{key}": statistics.median(r[kind][key] for r in rows) for kind in ("cold", "warm")
           for key in ("page_s", "ready_s", "first_word_s")}
    out = ROOT / "eval" / "results" / f"cold-start-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "median": med, "rows": rows}, indent=1))
    print(json.dumps(med, indent=1), "\nwrote", out)


if __name__ == "__main__":
    asyncio.run(main())
