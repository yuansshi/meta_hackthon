"""Spike S1: how fast do Meta's models start talking? Prints median / p90 per variant and saves JSON.

    uv run python scripts/probe.py --n 20         # Muse Spark variants, chaining check, Muse Spark Open if keyed
    uv run python scripts/probe.py --only asr     # realtime ASR round trip on speech synthesized by `say`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from openai import AsyncOpenAI

from earshot.config import ROOT, MissingKey, api_key, load_config, model_cfg
from earshot.relay import AsrRelay

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Current weather and today's forecast for a place.",
    "parameters": {"type": "object", "properties": {"place": {"type": "string"}},
                   "required": ["place"], "additionalProperties": False},
}
RAIN_Q = "Will it rain in Seattle this afternoon?"
VARIANTS = {
    "spark-minimal": ("minimal", "In one short sentence, suggest a name for a golden retriever.", []),
    "spark-low": ("low", "In one short sentence, suggest a name for a golden retriever.", []),
    "spark-tool": ("minimal", RAIN_Q, [WEATHER_TOOL]),
    "spark-search": ("minimal", "In one sentence: what is the latest news about Meta's Muse models?",
                     [{"type": "web_search"}]),
}


def client_for(m: dict) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=90, max_retries=0)


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


async def one_responses(client: AsyncOpenAI, m: dict, effort: str, prompt: str, tools: list) -> dict:
    t0 = time.perf_counter()
    first_text = first_tool = None
    final = None
    stream = await client.responses.create(model=m["id"], input=prompt, reasoning={"effort": effort},
                                           max_output_tokens=m["max_output_tokens"], stream=True,
                                           **({"tools": tools} if tools else {}))
    async for ev in stream:
        now = time.perf_counter() - t0
        if ev.type == "response.output_text.delta" and first_text is None:
            first_text = now
        elif ev.type == "response.output_item.added" and first_tool is None \
                and ev.item.type in ("function_call", "web_search_call"):
            first_tool = now
        elif ev.type == "response.completed":
            final = ev.response
    u = final.usage if final else None
    return {"first_text_s": first_text, "first_tool_s": first_tool, "total_s": time.perf_counter() - t0,
            "in": u.input_tokens if u else None, "out": u.output_tokens if u else None,
            "reasoning": u.output_tokens_details.reasoning_tokens if u else None,
            "text": final.output_text[:120] if final else None}


async def chain_test(client: AsyncOpenAI, m: dict, mode: str) -> dict:
    """Tool round trip: does `previous_response_id` (or resending history) work on this API?"""
    r1 = await client.responses.create(model=m["id"], input=RAIN_Q, tools=[WEATHER_TOOL],
                                       reasoning={"effort": "minimal"}, max_output_tokens=m["max_output_tokens"])
    calls = [o for o in r1.output if o.type == "function_call"]
    if not calls:
        return {"ok": False, "error": f"answered without the tool: {r1.output_text[:80]!r}"}
    result = {"type": "function_call_output", "call_id": calls[0].call_id,
              "output": json.dumps({"place": "Seattle", "now": "light rain, 52F", "afternoon_rain_chance_pct": 80})}
    if mode == "previous_response_id":
        extra = {"previous_response_id": r1.id, "input": [result]}
    else:
        prior = [o.model_dump(exclude_none=True, by_alias=True) for o in r1.output if o.type != "reasoning"]
        extra = {"input": [{"role": "user", "content": RAIN_Q}, *prior, result]}
    t0 = time.perf_counter()
    r2 = await client.responses.create(model=m["id"], tools=[WEATHER_TOOL], reasoning={"effort": "minimal"},
                                       max_output_tokens=m["max_output_tokens"], **extra)
    return {"ok": True, "s": round(time.perf_counter() - t0, 2), "answer": r2.output_text[:120]}


async def one_chat(client: AsyncOpenAI, m: dict, prompt: str) -> dict:
    t0 = time.perf_counter()
    first = None
    stream = await client.chat.completions.create(model=m["id"], messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=m["max_output_tokens"], stream=True,
                                                  reasoning_effort=m["reasoning_effort"])
    async for ch in stream:
        if first is None and ch.choices and ch.choices[0].delta.content:
            first = time.perf_counter() - t0
    return {"first_text_s": first, "total_s": time.perf_counter() - t0}


async def run_variant(name: str, n: int, call) -> dict:
    rows, errors = [], []
    for _ in range(n):
        try:
            rows.append(await call())
        except Exception as e:  # keep going: one bad call shouldn't hide the other variants
            errors.append(f"{type(e).__name__}: {e}"[:300])
    summary = {k: {"p50": pct([r.get(k) for r in rows], 0.5), "p90": pct([r.get(k) for r in rows], 0.9)}
               for k in ("first_text_s", "first_tool_s", "total_s", "reasoning")}
    print(f"{name:15} ok={len(rows):2}/{n}  first word p50={summary['first_text_s']['p50']}s "
          f"p90={summary['first_text_s']['p90']}s  tool p50={summary['first_tool_s']['p50']}s  "
          f"total p50={summary['total_s']['p50']}s  reasoning tok p50={summary['reasoning']['p50']}")
    if errors:
        print(f"{'':15} first error: {errors[0]}")
    if rows and rows[0].get("text"):
        print(f"{'':15} sample: {rows[0]['text']!r}")
    return {"summary": summary, "rows": rows, "errors": errors}


def synth_pcm(text: str, rate: int) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "probe.wav"
        subprocess.run(["say", "-o", str(path), "--file-format=WAVE", f"--data-format=LEI16@{rate}", text],
                       check=True)
        with wave.open(str(path)) as w:
            assert w.getframerate() == rate and w.getnchannels() == 1 and w.getsampwidth() == 2
            return w.readframes(w.getnframes())


async def asr_probe(cfg: dict, text: str) -> dict:
    a = cfg["asr"]
    pcm = synth_pcm(text, a["sample_rate"])
    events: list[tuple[float, dict]] = []
    done = asyncio.Event()

    async def on_event(evt: dict) -> None:
        events.append((time.perf_counter(), evt))
        if evt.get("type") == "speechComplete":
            done.set()

    relay = AsrRelay(a, api_key(a["api_key_env"]), on_event)
    task = asyncio.create_task(relay.run())
    t0 = time.perf_counter()
    while relay.session_id is None:
        if time.perf_counter() - t0 > 10:
            task.cancel()
            return {"ok": False, "error": f"no ASR session after 10 s; closes={relay.closes}"}
        await asyncio.sleep(0.05)
    open_s = time.perf_counter() - t0
    step = relay.frame_bytes
    for i in range(0, len(pcm), step):  # real-time pacing, as a phone would deliver it
        relay.feed(pcm[i:i + step])
        await asyncio.sleep(a["frame_ms"] / 1000)
    speech_end = time.perf_counter()
    try:
        await asyncio.wait_for(done.wait(), 15)
    except TimeoutError:
        pass
    await relay.close()
    task.cancel()
    complete = next(((t, e) for t, e in events if e.get("type") == "speechComplete"), None)
    out = {"ok": complete is not None, "handshake_s": round(open_s, 2), "said": text,
           "heard": complete[1].get("transcript") if complete else None,
           "speech_end_to_final_s": round(complete[0] - speech_end, 2) if complete else None,
           "event_types": sorted({e.get("type") for _, e in events}), "closes": relay.closes}
    print(json.dumps(out, indent=2))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only", choices=["spark", "open", "asr"])
    args = ap.parse_args()
    cfg = load_config()
    report: dict = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": args.n}

    if args.only in (None, "spark"):
        m = model_cfg(cfg, "muse-spark")
        client = client_for(m)
        print(f"== {m['id']} via {m['base_url']} (Responses API, {args.n} calls per variant)")
        for name, (effort, prompt, tools) in VARIANTS.items():
            report[name] = await run_variant(name, args.n, lambda e=effort, p=prompt, t=tools:
                                             one_responses(client, m, e, p, t))
        for mode in ("previous_response_id", "stateless"):
            try:
                report[f"chain-{mode}"] = r = await chain_test(client, m, mode)
            except Exception as e:
                report[f"chain-{mode}"] = r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"chain {mode:21} {r}")

    if args.only in (None, "open"):
        if os.environ.get("FIREWORKS_API_KEY"):
            g = model_cfg(cfg, "muse-spark-open")
            gclient = client_for(g)
            print(f"== {g['id']} via {g['base_url']} (Chat Completions, effort {g['reasoning_effort']})")
            report["muse-spark-open"] = await run_variant("muse-spark-open", args.n, lambda: one_chat(
                gclient, g, "In one short sentence, suggest a name for a golden retriever."))
        else:
            print("== muse-spark-open skipped: FIREWORKS_API_KEY not set")

    if args.only in (None, "asr"):
        print("== realtime ASR")
        report["asr"] = await asr_probe(cfg, "Hey Earshot, will it rain this afternoon?")

    out = ROOT / "results" / f"probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MissingKey as e:
        raise SystemExit(str(e))
