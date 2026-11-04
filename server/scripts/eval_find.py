"""Find it, for someone who can't see well: does Earshot say where the thing they asked about is?

    uv run python scripts/eval_find.py --n 60    # on the server (fly ssh console), which reaches Meta's image models

1. Muse Image makes a photo of an everyday scene with one target object on the left or the right.
2. Muse Spark, asked on its own, says which side the target is on. Where it agrees with the side Muse Image was
   asked for, that side is taken as the truth; the other photos count only for finding, not for position.
3. SAM 3.1 looks for the target (it's there) and for an object that isn't (it should find nothing).
4. The whole agent, through the running server's WebSocket, gets the photo and "Where is the …? Show me." It passes
   when the spoken answer names the right side and the app is told what to outline.
Writes eval/results/find-<date>.json. Photos and finished rows are kept under $EARSHOT_DATA_DIR/eval-find, so a
rerun (say, after the Fly trial stops the machine at its 5-minute mark) picks up where the last one stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from websockets.asyncio.client import connect

from earshot.config import DATA_DIR, api_key, load_config, model_cfg
from earshot.find import locate

ROOT = Path(__file__).resolve().parents[1]
# (scene, what to find, something that isn't there): everyday things someone with low vision might ask about
SCENES = [
    ("a hotel corridor", "exit sign", "bicycle"), ("a kitchen counter", "coffee mug", "traffic cone"),
    ("a living room", "TV remote", "fire hydrant"), ("a city street", "bus stop sign", "bathtub"),
    ("an office lobby", "elevator call button", "horse"), ("a wooden desk", "pair of glasses", "car"),
    ("a pharmacy aisle", "shopping basket", "piano"), ("a train platform", "departures board", "bed"),
    ("a park path", "trash can", "refrigerator"), ("a hallway", "fire extinguisher", "bicycle"),
    ("an entryway table", "set of keys", "guitar"), ("a supermarket shelf", "carton of milk", "laptop"),
    ("a restaurant table", "salt shaker", "bicycle"), ("a street crossing", "pedestrian signal", "bathtub"),
    ("a bathroom sink", "toothbrush", "car tire"), ("a coffee shop counter", "card reader", "ladder"),
    ("a bedroom nightstand", "alarm clock", "bicycle"), ("a parking garage", "parking ticket machine", "sofa"),
    ("a front door", "door handle", "piano"), ("a staircase", "handrail", "refrigerator"),
    ("a library shelf", "book with a red cover", "car"), ("a park bench", "water bottle", "television"),
    ("an airport gate area", "gate sign", "bathtub"), ("a laundry room", "laundry basket", "traffic light"),
    ("a garden", "watering can", "laptop"), ("a building entrance", "wheelchair ramp", "piano"),
    ("a hospital corridor", "restroom sign", "bicycle"), ("a kitchen", "microwave", "horse"),
    ("a living room", "potted plant", "bus"), ("a subway station", "ticket machine", "bed"),
]
# --hard: small things among clutter, in dim or uneven light: where someone with low vision most needs the help
HARD_SCENES = [
    ("a cluttered desk", "pair of scissors", "piano"), ("a crowded kitchen counter", "salt shaker", "bicycle"),
    ("a dim hallway at night", "light switch", "sofa"), ("a busy bus station", "ticket machine", "bathtub"),
    ("a messy living room", "TV remote", "fire hydrant"), ("a dark parking lot at night", "exit sign", "bed"),
    ("a cluttered entryway", "set of keys", "guitar"), ("a bathroom shelf of bottles", "bottle with a red cap", "car"),
    ("a foggy park", "bench", "refrigerator"), ("a staircase in low light", "handrail", "piano"),
    ("a crowded cafe", "power outlet", "horse"), ("a train platform at night", "platform number sign", "bathtub"),
    ("a hotel room in the evening", "thermostat", "bicycle"), ("an office kitchen", "coffee machine", "tractor"),
    ("a shop counter with many items", "card reader", "ladder"),
]
HARD = False
MIME = {b"\xff\xd8\xff": "jpeg", b"\x89PN": "png", b"RIF": "webp"}


def prompt(scene: str, target: str, side: str) -> str:
    if HARD:
        return (f"A realistic, everyday phone photo of {scene}, in dim or uneven light, with lots of other things "
                f"around. A small {target} is on the {side} side of the photo, partly among other objects, and there "
                f"is nothing like it on the other side. No people, no text overlays, no real brand names or logos.")
    return (f"A realistic, everyday phone photo of {scene}. A {target} is clearly visible on the {side} side of the "
            f"photo, and there is nothing like it on the other side. Natural light. No people, no text overlays, "
            f"no real brand names or logos.")


async def make_photo(client: AsyncOpenAI, path: Path, text: str, sem: asyncio.Semaphore) -> tuple[str, float]:
    if not path.exists():
        async with sem:
            t = time.perf_counter()
            img = await client.images.generate(model="muse-image-1.0", prompt=text, n=1)
            first = img.data[0]
            data = (base64.b64decode(first.b64_json) if getattr(first, "b64_json", None)
                    else httpx.get(first.url, timeout=60).content)
            path.write_bytes(data)
            seconds = time.perf_counter() - t
    else:
        seconds = 0.0
    data = path.read_bytes()
    return f"data:image/{MIME.get(data[:3], 'png')};base64," + base64.b64encode(data).decode(), seconds


async def spark_side(client: AsyncOpenAI, url: str, target: str) -> tuple[str, float]:
    r = await client.responses.create(
        model="muse-spark-1.3", reasoning={"effort": "minimal"}, max_output_tokens=600,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": f"In this photo, is the {target} in the left half or the right half? "
                                           "Reply with exactly one word: left, right, or none."},
            {"type": "input_image", "image_url": url}]}])
    word = (r.output_text or "").strip().lower().strip(".")
    u = r.usage
    cost = (u.input_tokens * 1.25 + u.output_tokens * 4.25) / 1e6 if u else 0.0
    return (word if word in ("left", "right", "none") else "unclear"), cost


def side_of(b: dict) -> str:
    return "left" if (b["x1"] + b["x2"]) / 2 < 0.5 else "right"


async def ask_agent(url: str, image: str, target: str, sem: asyncio.Semaphore) -> dict:
    """The whole loop, as a judge would use it: photo plus question, spoken answer, outline."""
    async with sem:
        async with connect(url, open_timeout=30, max_size=None) as ws:
            await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "ptt", "device_id": f"eval-{uuid.uuid4().hex[:8]}",
                                      "tz": "America/Los_Angeles", "sample_rate": 16000, "tools": []}))
            sent, said, first, highlight, done = None, [], None, None, None
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "ready":
                    sent = time.perf_counter()
                    await ws.send(json.dumps({"type": "image", "data": image, "text": f"Where is the {target}? Show me."}))
                elif msg["type"] == "say" and msg.get("text"):
                    first = first or time.perf_counter()
                    said.append(msg["text"])
                elif msg["type"] == "highlight":
                    highlight = msg["boxes"]
                elif msg["type"] == "turn_done":
                    done = msg
                    break
            words = " ".join(said)
            return {"said": words, "highlight": highlight, "first_s": round(first - sent, 2) if first else None,
                    "total_s": round(time.perf_counter() - sent, 2), "cost": (done or {}).get("cost_usd"),
                    "tools": (done or {}).get("tools")}


def says_side(text: str, side: str) -> bool:
    t = text.lower()
    other = "right" if side == "left" else "left"
    return side in t and other not in t


async def one(i: int, scene: tuple, side: str, client, ws_url: str, sems: dict) -> dict:
    place, target, absent = scene
    photo_dir = DATA_DIR / ("eval-find-hard" if HARD else "eval-find")
    photo_dir.mkdir(parents=True, exist_ok=True)
    row = {"i": i, "scene": place, "target": target, "absent": absent, "asked_side": side}
    try:
        url, row["image_s"] = await make_photo(client, photo_dir / f"{i:03d}-{side}.img", prompt(place, target, side),
                                               sems["image"])
        async with sems["models"]:
            (row["spark_side"], row["spark_cost"]), (found, row["sam_s"]), (ghost, _) = await asyncio.gather(
                spark_side(client, url, target), locate(client, "sam-3.1", url, target),
                locate(client, "sam-3.1", url, absent))
        row["truth"] = side if row["spark_side"] == side else None
        row["sam_found"] = len(found)
        row["sam_side"] = side_of(max(found, key=lambda b: b["c"] or 0)) if found else None
        row["sam_absent_found"] = len(ghost)
        e2e = await asyncio.wait_for(ask_agent(ws_url, url, target, sems["agent"]), 120)
        row.update({f"agent_{k}": v for k, v in e2e.items()})
        if row["truth"]:
            row["agent_side_ok"] = says_side(e2e["said"], row["truth"])
    except Exception as e:  # one bad photo shouldn't sink the run
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def pct(rows: list, key: str, test) -> str:
    rs = [r for r in rows if key in r and r[key] is not None]
    return f"{sum(bool(test(r)) for r in rs)}/{len(rs)}" if rs else "0/0"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="photos: each scene once on the left, once on the right")
    ap.add_argument("--ws", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--hard", action="store_true", help="small things in clutter and dim light")
    a = ap.parse_args()
    global HARD, SCENES
    if a.hard:
        HARD, SCENES = True, HARD_SCENES
    cfg = load_config()
    m = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=180, max_retries=2)
    items = [(i, SCENES[i % len(SCENES)], "left" if (i // len(SCENES)) % 2 == 0 else "right") for i in range(a.n)]
    sems = {"image": asyncio.Semaphore(6), "models": asyncio.Semaphore(6), "agent": asyncio.Semaphore(4)}
    t0 = time.perf_counter()
    saved = DATA_DIR / ("eval-find-hard" if HARD else "eval-find") / "rows.jsonl"
    saved.parent.mkdir(parents=True, exist_ok=True)
    have = {r["i"]: r for r in map(json.loads, saved.read_text().splitlines()) if "error" not in r} if saved.exists() else {}

    async def kept(i, scene, side):
        if i in have:
            return have[i]
        row = await one(i, scene, side, client, a.ws, sems)
        if "error" not in row:
            with saved.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    rows = await asyncio.gather(*(kept(i, scene, side) for i, scene, side in items))
    ok = [r for r in rows if "error" not in r]
    truth = [r for r in ok if r.get("truth")]
    firsts = [r["agent_first_s"] for r in ok if r.get("agent_first_s")]
    totals = [r["agent_total_s"] for r in ok if r.get("agent_total_s")]
    summary = {
        "photos": len(ok), "errors": len(rows) - len(ok),
        "truth_side_agreed": f"{len(truth)}/{len(ok)}",
        "sam_found_target": pct(ok, "sam_found", lambda r: r["sam_found"] > 0),
        "sam_found_nothing_for_absent_object": pct(ok, "sam_absent_found", lambda r: r["sam_absent_found"] == 0),
        "sam_right_side": pct(truth, "sam_side", lambda r: r["sam_side"] == r["truth"]),
        "agent_said_right_side": pct(truth, "agent_side_ok", lambda r: r["agent_side_ok"]),
        "agent_outlined": pct(ok, "agent_said", lambda r: r.get("agent_highlight")),
        "agent_first_word_p50_s": statistics.median(firsts) if firsts else None,
        "agent_total_p50_s": statistics.median(totals) if totals else None,
        "agent_cost_per_question_usd": round(statistics.mean([r["agent_cost"] for r in ok if r.get("agent_cost")]), 4)
        if ok else None,
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    out = DATA_DIR / ("eval-find-hard" if HARD else "eval-find") / f"find-{time.strftime('%Y-%m-%d')}.json"  # on the volume
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if "error" in r or (r.get("truth") and not r.get("agent_side_ok")):
            print("MISS", r["i"], r["target"], r.get("truth"), r.get("sam_side"), r.get("error", ""), "|",
                  r.get("agent_said", "")[:100])
    print("wrote", out)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
