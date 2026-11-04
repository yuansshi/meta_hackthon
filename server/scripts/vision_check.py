"""Spike: can each brain see a photo? Sends a two-color test image in the same shape the agent uses.

    uv run python scripts/vision_check.py      # Muse Spark (Responses API) and Muse Spark Open (Chat)

Run it on the Fly machine for Meta: the Mac's network gets 404s from Meta's model endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import struct
import time
import zlib

from openai import AsyncOpenAI

from everymuse.config import MissingKey, api_key, load_config, model_cfg

QUESTION = "This test image has two halves. Name the color of the left half and the right half, in five words."


def two_color_png(w: int = 96, h: int = 96) -> bytes:
    """A left-red, right-blue PNG built by hand, so the check needs no imaging library."""
    row = b"\x00" + b"".join(b"\xe0\x20\x20" if x < w // 2 else b"\x20\x40\xe0" for x in range(w))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))


async def ask(m: dict, image_url: str) -> str:
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=90, max_retries=0)
    if m["api"] == "chat":
        r = await client.chat.completions.create(
            model=m["id"], max_tokens=800, reasoning_effort=m["reasoning_effort"],
            messages=[{"role": "user", "content": [{"type": "text", "text": QUESTION},
                                                   {"type": "image_url", "image_url": {"url": image_url}}]}])
        return r.choices[0].message.content or ""
    r = await client.responses.create(
        model=m["id"], max_output_tokens=800, reasoning={"effort": m["reasoning_effort"]},
        input=[{"role": "user", "content": [{"type": "input_text", "text": QUESTION},
                                            {"type": "input_image", "image_url": image_url}]}])
    return r.output_text


async def main() -> None:
    cfg = load_config()
    image_url = "data:image/png;base64," + base64.b64encode(two_color_png()).decode()
    for key in ("muse-spark", "muse-spark-open"):
        m = model_cfg(cfg, key)
        t0 = time.perf_counter()
        try:
            answer = await ask(m, image_url)
            print(f"{key:16} {time.perf_counter() - t0:5.2f}s  {answer.strip()!r}")
        except MissingKey as e:
            print(f"{key:16} skipped: {e}")
        except Exception as e:
            print(f"{key:16} ERROR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
