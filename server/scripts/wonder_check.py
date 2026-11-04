"""Spike: does Muse Wonder run as a background response on Meta's API, and how long does a small job take?

    uv run python -u scripts/wonder_check.py     # run on the Fly machine: the Mac's network gets 404s from Meta
"""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

from everymuse.config import api_key, load_config, model_cfg

JOB = ("Find the three tallest buildings in Seattle and their heights. Reply as SUMMARY: one spoken sentence. "
       "DETAILS: a short list with sources.")


async def main() -> None:
    m = model_cfg(load_config(), "muse-wonder")
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=60, max_retries=0)
    t0 = time.perf_counter()
    try:
        r = await client.responses.create(model=m["id"], input=JOB, tools=[{"type": "web_search"}],
                                          reasoning={"effort": "low"}, max_output_tokens=6000, background=True)
    except Exception as e:
        print("background create failed:", type(e).__name__, str(e)[:300])
        return
    print(f"created {r.id} status={r.status} after {time.perf_counter() - t0:.1f}s", flush=True)
    while r.status in ("queued", "in_progress"):
        await asyncio.sleep(3)
        r = await client.responses.retrieve(r.id)
        print(f"  {time.perf_counter() - t0:5.1f}s status={r.status}", flush=True)
    u = r.usage
    searches = sum(1 for o in r.output if o.type == "web_search_call")
    print(f"done: {r.status} after {time.perf_counter() - t0:.1f}s; tokens in={u.input_tokens if u else '?'} "
          f"out={u.output_tokens if u else '?'}; web searches={searches}")
    print(r.output_text[:900])


if __name__ == "__main__":
    asyncio.run(main())
