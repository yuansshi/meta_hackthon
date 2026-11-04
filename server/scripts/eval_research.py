"""Does "look into it and get back to me" hold up beyond one demo? 20 varied research jobs through EveryMuse's own
Muse Wonder path (background mode, polled), as the agent would hand them over.

    uv run python scripts/eval_research.py --jobs 20    # on the server (fly ssh console), which reaches Meta's API

Saves each finished job to $EVERYMUSE_DATA_DIR/eval-research/ as it goes, so a rerun after a trial restart resumes.
Records for each job whether a report came back, how long it took, how many web searches it ran, what it cost,
and the spoken summary. Writes $EVERYMUSE_DATA_DIR/eval-research.json (the server's volume) and prints a summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from everymuse.actions import ActionLog
from everymuse.budget import Budget
from everymuse.config import DATA_DIR, load_config
from everymuse.research import Job, Researcher, split_report

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
GOALS += ['the best carry-on suitcases under $250', 'how to get from Heathrow to central London cheaply', 'kid-friendly hikes near Portland', 'whether Pike Place Market is open on Thanksgiving', 'the quietest dishwashers under $800', "how to renew a Washington driver's license online", 'gluten-free bakeries in Capitol Hill, Seattle', 'the best budget mechanical keyboards', 'rules for flying with a power bank', 'free museum days in Seattle this month', 'how to get a library card at Seattle Public Library', 'the best rain jackets for cycling', 'dog-friendly cafes in Ballard', 'how to recycle electronics in King County', 'the cheapest way to send a parcel to Canada', 'beginner yoga studios near downtown Seattle', 'the best e-readers for reading in sunlight', "Seattle's ferry schedule to Bainbridge on weekends", 'how to dispute a parking ticket in Seattle', 'vegetarian meal kits that deliver to Seattle', 'the best travel adapters for Europe', 'how to get a TSA PreCheck appointment', 'affordable piano lessons for adults in Seattle', 'the best indoor plants that are safe for cats', "how long the Space Needle's lines are on weekends", 'the best noise machines for sleep', 'how to get to Mount Rainier without a car', 'farmers markets open in Seattle in November', 'the safest booster seats for kids', 'how to apply for a Washington State Parks Discover Pass', 'the best budget standing desks', 'Seattle neighborhoods with good coffee shops for working', 'how to set up two-factor authentication on Gmail', 'the best running routes around Green Lake', 'where to donate used clothes in Seattle', 'the best rice cookers under $150', 'how to get a passport photo taken near Seattle', 'late-night pharmacies in Seattle', 'the best bike locks for city commuting', 'how to transfer a phone number to a new carrier', 'family board games for ages 8 and up', 'how to check Seattle air quality today', 'the best water bottles that keep drinks cold', 'cheap parking near Lumen Field on game days', 'how to clean a cast iron pan', 'the best headphones for open-plan offices', 'sledding spots within two hours of Seattle', 'how to book a campsite at Deception Pass', 'the best portable chargers for phones', 'how to report a pothole in Seattle', 'the best compact umbrellas', 'how to freeze fresh herbs', 'the best laptop backpacks for commuting by bike', 'Seattle libraries with study rooms you can book', 'how to get a refund from Ticketmaster', 'the best air purifiers for wildfire smoke', 'how to winterize a balcony garden', "the best kids' rain boots", 'how to find a notary in Seattle', 'the best budget webcams for video calls', 'where to see the northern lights near Seattle this week', 'how to back up an iPhone to a Mac', 'the best electric kettles', 'how to get to Snoqualmie Pass by bus', 'the best meal-prep containers', 'how to sign up for Seattle City Light budget billing', 'the best gloves for using a phone in the cold', 'kid-friendly restaurants in Fremont, Seattle', 'how to set up a new router', 'the best walking shoes for travel', 'how to recycle styrofoam in Seattle', 'the best cordless vacuums under $300', 'how to replace a lost Orca card', 'the best books for learning to cook', 'how to get a fishing license in Washington', 'the best sunrise viewpoints in Seattle', 'how to make a reservation at the Seattle Aquarium', 'the best travel pillows for long flights', 'how to file a lost item claim with Sound Transit', 'the best headlamps for winter running']


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=100)
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
    store = DATA_DIR / "eval-research"
    store.mkdir(parents=True, exist_ok=True)
    saved = store / "rows.jsonl"
    have = {r["goal"]: r for r in map(json.loads, saved.read_text().splitlines())} if saved.exists() else {}

    async def kept(k: int, goal: str) -> dict:
        if goal in have and have[goal].get("ok"):
            return have[goal]
        row = await one(k, goal)
        if row["ok"]:
            with saved.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    try:
        rows = await asyncio.wait_for(asyncio.gather(*(kept(k, g) for k, g in enumerate(GOALS[:a.jobs]))), 255)
    except TimeoutError:
        print("not finished: run again")
        return
    ok = [x for x in rows if x["ok"]]
    summary = {"jobs": len(rows), "reported": len(ok), "models": sorted({x.get("model", "") for x in ok}),
               "seconds_p50": statistics.median(x["s"] for x in ok) if ok else None,
               "seconds_max": max((x["s"] for x in ok), default=None),
               "searches_p50": statistics.median(x["searches"] for x in ok) if ok else None,
               "cost_p50": statistics.median(x["cost"] for x in ok) if ok else None,
               "cost_total": round(sum(x.get("cost", 0) for x in rows), 3), "wall_s": round(time.perf_counter() - t0, 1)}
    out = store / "summary.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for x in rows:
        print("OK  " if x["ok"] else "FAIL", f"{x['s']:6.1f}s", x.get("searches"), x["goal"], "|", (x.get("summary") or x.get("error", ""))[:110])
    print("done" if all(x["ok"] for x in rows) else "not finished: run again")


if __name__ == "__main__":
    asyncio.run(main())
