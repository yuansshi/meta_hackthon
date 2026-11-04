"""An hour in a pocket: one iPhone-style session left open past speech recognition's 60-minute session cap.

    uv run python scripts/eval_soak.py --url wss://earshot-yuansshi.fly.dev/ws --minutes 66 --every 5

The session stays open the whole time, as the iPhone app's does; between prompts the client sends nothing and the
server's relay fills silence on its own clock. Every few minutes the harness says "Hey EveryMuse, what time is it?"
and records whether it was heard and answered, and how fast. Muse Voice Transcribe closes a session at 60 minutes,
so the prompts after that show whether the relay reconnected without the user noticing.
Writes eval/results/soak-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_live import Trial, done  # noqa: E402
from probe import synth_pcm  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://earshot-yuansshi.fly.dev/ws")
    ap.add_argument("--minutes", type=float, default=66)
    ap.add_argument("--every", type=float, default=5)
    a = ap.parse_args()
    pcm = synth_pcm("Hey EveryMuse, what time is it?", 16000)
    out = ROOT / "eval" / "results" / f"soak-{time.strftime('%Y-%m-%d')}.json"
    rows: list[dict] = []
    started = time.perf_counter()
    async with connect(a.url, open_timeout=60, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "ios", "mode": "wake", "device_id": f"soak-{uuid.uuid4().hex[:8]}",
                                  "tz": "America/Los_Angeles", "sample_rate": 16000,
                                  "tools": ["get_location", "create_reminder"]}))
        tr = Trial(ws, started)
        reader = asyncio.create_task(tr.read())
        await tr.wait(lambda m: m.get("type") == "ready", 30)
        await asyncio.sleep(3)
        k = 0
        while (elapsed := (time.perf_counter() - started) / 60) < a.minutes:
            end = await tr.speak(pcm)
            row = {"n": k, "minute": round(elapsed, 1)}
            try:
                t, _ = await tr.wait(done, 40, end)
                turn = tr.turn(end, t)
                heard = [m for tt, m in tr.events if tt >= end - 5 and m.get("type") == "transcript" and m.get("final")]
                row.update(ok=bool(turn["said"]), first_s=turn["first_s"], said=turn["said"][:120],
                           heard=heard[-1]["text"] if heard else "")
            except TimeoutError:
                row.update(ok=False, first_s=None, said="", heard="")
            rows.append(row)
            print(json.dumps(row), flush=True)
            out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows}, indent=1))
            k += 1
            await asyncio.sleep(max(0.0, a.every * 60 - (time.perf_counter() - end)))
        reader.cancel()
    ok = sum(r["ok"] for r in rows)
    after = [r for r in rows if r["minute"] > 60]
    summary = {"prompts": len(rows), "answered": ok, "after_60_min": f"{sum(r['ok'] for r in after)}/{len(after)}",
               "minutes": round((time.perf_counter() - started) / 60, 1)}
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary), "\nwrote", out)


if __name__ == "__main__":
    asyncio.run(main())
