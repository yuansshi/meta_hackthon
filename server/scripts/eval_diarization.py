"""Can Muse Voice Transcribe tell who said what? Diarization on synthetic two-person conversations.

    uv run python scripts/eval_diarization.py --conversations 12

Heads-up's remaining mistakes are other people's plans: the text alone doesn't say whose plan it is. This measures
whether the diarization mode could supply that. Each conversation alternates two macOS voices over 4 to 6 lines with
short pauses, and streams at real speed in DIARIZATION mode. Measured:
- turns: how many finished turns came back, against the lines spoken (merged lines show up as fewer turns);
- attribution: the share of turns whose speaker label matches who really spoke, under the best label-to-speaker map;
- latency: end of the line to its finished turn.
Writes eval/results/diarization-<date>.json. Speech recognition works from this Mac.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from eval_languages import RATE, pcm_of, silence, stream

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("Alex", "Daniel"), ("Samantha", "Rishi"), ("Karen", "Alex"), ("Moira", "Tessa"), ("Daniel", "Samantha"),
         ("Tessa", "Alex")]
LINES = [
    ("Are you coming to the team lunch on Thursday?", "I think so. Is it still at the noodle place on Fifth?"),
    ("Yes, twelve thirty. Can you bring the projector cable?", "Sure, I'll grab it from the supply room."),
    ("My sister lands at six tomorrow, so I'll be late.", "No problem, we'll start without you."),
    ("Did you see the game last night?", "Only the last quarter, but what a finish."),
    ("The plumber said between two and four on Friday.", "Then someone needs to be home all afternoon."),
    ("Don't forget your passport for the Lisbon trip.", "It's already in my bag, next to the tickets."),
]


async def one(k: int, work: Path, sem: asyncio.Semaphore) -> dict:
    rnd = random.Random(k)
    a, b = PAIRS[k % len(PAIRS)]
    picks = rnd.sample(LINES, 3)
    script = [(who, line) for pair in picks for who, line in zip((a, b), pair)][: 4 + k % 3]
    parts, ends, at = [silence(0.5)], [], 0.5
    for i, (voice, line) in enumerate(script):
        pcm = pcm_of(voice, line, work, f"c{k}-{i}")
        parts.append(pcm)
        at += len(pcm) / 2 / RATE
        ends.append((voice, at))
        gap = rnd.choice((0.6, 0.9, 1.2))
        parts.append(silence(gap))
        at += gap
    async with sem:
        t0, events = await stream(b"".join(parts) + silence(3.0), "DIARIZATION", ["English"])
    turns, label = [], None
    for t, e in events:
        if e.get("type") == "speechStart":
            label = None
        elif e.get("type") == "speaker":
            label = e.get("label")
        elif e.get("type") == "speechComplete" and e.get("transcript"):
            at_s = t - t0
            before = [x for x in ends if x[1] <= at_s + 0.05]
            truth = before[-1][0] if before else None
            turns.append({"label": label, "truth": truth, "latency_s": round(at_s - before[-1][1], 2) if before else None,
                          "text": e["transcript"]})
    labels = sorted({t["label"] for t in turns if t["label"]})
    best = 0
    for perm in itertools.permutations(labels, min(2, len(labels))):  # the label-to-speaker map that fits best
        m = dict(zip(perm, (a, b)))
        best = max(best, sum(1 for t in turns if m.get(t["label"]) == t["truth"]))
    return {"conversation": k, "voices": [a, b], "lines": len(script), "turns": len(turns), "labels": labels,
            "attributed": best, "latency": [t["latency_s"] for t in turns if t["latency_s"] is not None], "detail": turns}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=12)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="diar-"))
    rows = await asyncio.gather(*(one(k, work, asyncio.Semaphore(6)) for k in range(a.conversations)))
    turns, lines = sum(r["turns"] for r in rows), sum(r["lines"] for r in rows)
    lat = [x for r in rows for x in r["latency"]]
    summary = {"conversations": len(rows), "lines": lines, "turns": turns,
               "attributed": f"{sum(r['attributed'] for r in rows)}/{turns}",
               "two_labels": f"{sum(len(r['labels']) == 2 for r in rows)}/{len(rows)}",
               "latency_p50_s": statistics.median(lat) if lat else None, "latency_p90_s": sorted(lat)[int(0.9 * len(lat))] if lat else None}
    out = ROOT / "eval" / "results" / f"diarization-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["conversation"], r["voices"], f"lines {r['lines']} turns {r['turns']} labels {r['labels']} right {r['attributed']}")
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
