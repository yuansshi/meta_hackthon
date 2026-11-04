"""Several judges at once: does the 512 MB server stay quick with 1, 4 and 8 sessions talking at the same time?

    uv run python scripts/eval_load.py --url wss://earshot-yuansshi.fly.dev/ws --levels 1,4,8

Each session types four requests in a row, as a judge trying the demo would: the weather (location and forecast
tools), remember where I parked, where did I park, and a web-search question. Device tools get canned answers, as
in eval_live.py. 8 is the server's limit of open sessions per network address. Writes
eval/results/load-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_live import Trial, done  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Will it rain this afternoon?", "Remember I parked on level 2, spot 17.", "Where did I park?",
           "What's the latest news about Meta's Muse models?"]


async def session(url: str) -> list[dict]:
    async with connect(url, open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "ptt", "device_id": f"load-{uuid.uuid4().hex[:8]}",
                                  "tz": "America/Los_Angeles", "sample_rate": 16000,
                                  "tools": ["get_location", "create_reminder"]}))
        tr = Trial(ws, time.perf_counter())
        reader = asyncio.create_task(tr.read())
        try:
            await tr.wait(lambda m: m.get("type") == "ready", 30)
            out = []
            for p in PROMPTS:
                sent = await tr.text(p)
                try:
                    t, _ = await tr.wait(done, 60, sent)
                    out.append({"prompt": p, **tr.turn(sent, t), "ok": True})
                except TimeoutError:
                    out.append({"prompt": p, "ok": False, "first_s": None, "total_s": None, "cost": 0})
                await asyncio.sleep(0.5)
            return out
        finally:
            reader.cancel()


def q(xs: list[float], p: float):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://earshot-yuansshi.fly.dev/ws")
    ap.add_argument("--levels", default="1,4,8")
    a = ap.parse_args()
    result = {}
    for n in map(int, a.levels.split(",")):
        t0 = time.perf_counter()
        runs = await asyncio.gather(*(session(a.url) for _ in range(n)), return_exceptions=True)
        turns = [t for r in runs if isinstance(r, list) for t in r]
        errors = [f"{type(r).__name__}: {r}"[:120] for r in runs if isinstance(r, BaseException)]
        firsts = [t["first_s"] for t in turns if t.get("first_s") is not None]
        result[str(n)] = {"sessions": n, "turns": len(turns), "ok": sum(t["ok"] for t in turns),
                          "session_errors": errors, "first_word_p50_s": q(firsts, 0.5), "first_word_p90_s": q(firsts, 0.9),
                          "turn_p50_s": q([t.get("total_s") for t in turns], 0.5),
                          "turn_p90_s": q([t.get("total_s") for t in turns], 0.9),
                          "cost_usd": round(sum(t.get("cost") or 0 for t in turns), 4),
                          "wall_s": round(time.perf_counter() - t0, 1), "rows": turns}
        print(n, {k: v for k, v in result[str(n)].items() if k != "rows"})
        await asyncio.sleep(3)
    out = ROOT / "eval" / "results" / f"load-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": a.url, "levels": result}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
