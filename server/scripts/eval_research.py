"""Does "look into it and get back to me" hold up beyond one demo? 20 varied research jobs through Earshot's own
Muse Wonder path (background mode, polled), as the agent would hand them over.

    uv run python scripts/eval_research.py --jobs 20    # on the server (fly ssh console), which reaches Meta's API

Records for each job whether a report came back, how long it took, how many web searches it ran, what it cost,
and the spoken summary. Writes $EARSHOT_DATA_DIR/eval-research.json (the server's volume) and prints a summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from earshot.actions import ActionLog
from earshot.budget import Budget
from earshot.config import DATA_DIR, load_config
from earshot.research import Job, Researcher, split_report

GOALS = [
    "the best noise-cancelling earbuds under $200", "which Seattle libraries are open late on Fridays",
    "the cheapest way to get from SFO to downtown San Francisco", "beginner-friendly day hikes within an hour of Seattle",
    "whether a US citizen needs a visa for a week in Japan", "free kid-friendly museums in Chicago on weekdays",
    "how to recycle old phone batteries in Seattle", "vegetarian restaurants near Pike Place Market open after 9 pm",
    "the best time of year to visit Lisbon on a budget", "compact strollers that fold with one hand under $300",
    "how to get a refund for a delayed Amtrak train", "electric toothbrushes dentists recommend under $100",
    "the easiest houseplants for a dark apartment", "what documents I need to renew a US passport by mail",
    "budget weekend trips from Seattle by train", "running shoes for flat feet under $150",
    "setting up mesh Wi-Fi for a three-bedroom house", "the rules for bringing a dog on Washington State Ferries",
    "the best free apps for learning Spanish", "where to see fall colors near Portland this week",
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=230)
    a = ap.parse_args()
    cfg = load_config()
    r = Researcher(cfg, Budget(1000))
    sem = asyncio.Semaphore(10)

    async def one(k: int, goal: str) -> dict:
        job = Job(f"eval{k}", "eval", goal, "The user is in Seattle.", "America/Los_Angeles", ActionLog(None))
        async with sem:
            t = time.perf_counter()
            try:
                text, model, searches, cost = await asyncio.wait_for(r._ask(job), a.timeout)
                summary, details = split_report(text)
                return {"goal": goal, "ok": bool(summary), "s": round(time.perf_counter() - t, 1), "model": model,
                        "searches": searches, "cost": round(cost, 4), "summary": summary, "details_chars": len(details)}
            except Exception as e:
                return {"goal": goal, "ok": False, "s": round(time.perf_counter() - t, 1), "error": f"{type(e).__name__}: {e}"[:160]}

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(one(k, g) for k, g in enumerate(GOALS[:a.jobs])))
    ok = [x for x in rows if x["ok"]]
    summary = {"jobs": len(rows), "reported": len(ok), "models": sorted({x.get("model", "") for x in ok}),
               "seconds_p50": statistics.median(x["s"] for x in ok) if ok else None,
               "seconds_max": max((x["s"] for x in ok), default=None),
               "searches_p50": statistics.median(x["searches"] for x in ok) if ok else None,
               "cost_p50": statistics.median(x["cost"] for x in ok) if ok else None,
               "cost_total": round(sum(x.get("cost", 0) for x in rows), 3), "wall_s": round(time.perf_counter() - t0, 1)}
    out = DATA_DIR / "eval-research.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for x in rows:
        print("OK  " if x["ok"] else "FAIL", f"{x['s']:6.1f}s", x.get("searches"), x["goal"], "|", (x.get("summary") or x.get("error", ""))[:110])


if __name__ == "__main__":
    asyncio.run(main())
