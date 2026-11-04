"""Heads-up end to end, by voice: the whole cascade on the live server, not just the screener on text.

    uv run python scripts/eval_headsup_spoken.py --per-category 8

Snippets come from the cross-checked hard set (`eval/hard2/`, labels both Muse Wonder passes agreed on), the same
number from each of the 12 categories. For each one, in its own hands-free session with Heads-up on:
1. the user's notes, if the snippet has any, are saved first ("Remember: …", typed), as a user would have;
2. the snippet's lines are spoken in two alternating macOS voices, streamed in real time like a phone's mic;
3. it passes when EveryMuse makes an offer for a snippet that deserves one, or stays quiet (12 s) for one that doesn't.
That covers speech recognition, the turn gate, the Rewind buffer, the debounce, the screener and the offer itself.
Writes eval/results/headsup-spoken-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import tempfile
import time
import uuid
import wave
from collections import Counter, defaultdict
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
URL = "wss://earshot-yuansshi.fly.dev/ws"
RATE = 16000
VOICES = [("Daniel", "Karen"), ("Moira", "Alex"), ("Tessa", "Daniel"), ("Rishi", "Samantha")]


def pcm_of(voice: str, text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "a.aiff", Path(d) / "a.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(RATE), "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return w.readframes(w.getnframes())


def snippets(per_category: int) -> list[dict]:
    snips = {json.loads(x)["id"]: json.loads(x) for x in open(ROOT / "eval" / "hard2" / "snippets.jsonl")}
    judged = {json.loads(x)["id"]: json.loads(x)["judge"] for x in open(ROOT / "eval" / "hard2" / "judged.jsonl")}
    by = defaultdict(list)
    for i, s in snips.items():
        if judged.get(i) == s["expect"]:
            by[s["category"]].append(s)
    rnd = random.Random(11)
    return [s for c in sorted(by) for s in rnd.sample(by[c], min(per_category, len(by[c])))]


async def one(k: int, s: dict, gate: asyncio.Semaphore) -> dict:
    a, b = VOICES[k % len(VOICES)]
    audio = b"".join(pcm_of(a if i % 2 == 0 else b, line) + b"\0\0" * int(RATE * 0.6) for i, line in enumerate(s["heard"]))
    row = {"id": s["id"], "category": s["category"], "expect": s["expect"]}
    async with gate:
        for attempt in range(3):  # the trial server restarts every 5 minutes: a dropped session is retried
            try:
                async with connect(URL, open_timeout=60, max_size=None) as ws:
                    await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "wake", "heads_up": True,
                                              "device_id": f"spoken-{uuid.uuid4().hex[:8]}", "tz": "America/Los_Angeles",
                                              "sample_rate": RATE, "tools": []}))
                    events: list[tuple[float, dict]] = []

                    async def read():
                        async for raw in ws:
                            events.append((time.perf_counter(), json.loads(raw)))
                    reader = asyncio.create_task(read())
                    for _ in range(300):
                        if any(m.get("type") == "ready" for _, m in events):
                            break
                        await asyncio.sleep(0.1)
                    for note in s["notes"]:  # what the user had saved before
                        n = len([m for _, m in events if m.get("type") == "turn_done"])
                        await ws.send(json.dumps({"type": "text", "text": f"Remember: {note}"}))
                        for _ in range(300):
                            if len([m for _, m in events if m.get("type") == "turn_done"]) > n:
                                break
                            await asyncio.sleep(0.1)
                    await asyncio.sleep(1.5)  # speech recognition opens
                    for i in range(0, len(audio), 640):
                        await ws.send(audio[i:i + 640])
                        await asyncio.sleep(0.02)
                    end = time.perf_counter()
                    offer = None
                    while time.perf_counter() - end < 12:
                        hit = [m for _, m in events if m.get("type") in ("confirm", "say") and str(m.get("turn", "")).startswith("h")
                               and (m.get("type") == "confirm" or m.get("text"))]
                        if hit:
                            offer = hit[0].get("question") or hit[0].get("text")
                            row["offer_s"] = round([t for t, m in events if m is hit[0]][0] - end, 2)
                            break
                        await asyncio.sleep(0.2)
                    row["heard"] = " / ".join(m["text"] for _, m in events if m.get("type") == "transcript" and m.get("final"))[:300]
                    row["offered"], row["offer"] = offer is not None, offer
                    reader.cancel()
                    row["ok"] = row["offered"] == (s["expect"] == "offer")
                    return row
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:120]
                await asyncio.sleep(5)
    row["ok"] = None
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=8)
    a = ap.parse_args()
    items = snippets(a.per_category)
    gate = asyncio.Semaphore(3)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*(one(k, s, gate) for k, s in enumerate(items)))
    done = [r for r in rows if r["ok"] is not None]
    tp = sum(r["offered"] and r["expect"] == "offer" for r in done)
    fp = sum(r["offered"] and r["expect"] == "none" for r in done)
    pos = sum(r["expect"] == "offer" for r in done)
    lat = sorted(r["offer_s"] for r in done if r.get("offer_s") is not None and r["expect"] == "offer")
    summary = {"snippets": len(items), "completed": len(done), "precision": round(tp / (tp + fp), 3) if tp + fp else None,
               "recall": round(tp / pos, 3) if pos else None, "false_offers": fp,
               "false_offers_by_category": dict(Counter(r["category"] for r in done if r["offered"] and r["expect"] == "none")),
               "missed_by_category": dict(Counter(r["category"] for r in done if not r["offered"] and r["expect"] == "offer")),
               "offer_after_speech_p50_s": lat[len(lat) // 2] if lat else None, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    out = ROOT / "eval" / "results" / f"headsup-spoken-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
