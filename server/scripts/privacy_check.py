"""Privacy filter check: Muse Image makes a synthetic street photo with people in it, SAM 3.1 finds the faces
(and one object), and the boxes come back for blurring. Prints timings and costs, and writes the photo and boxes.

    uv run python scripts/privacy_check.py --out /tmp/privacy     # needs a network that reaches Meta's model API
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from everymuse.config import api_key, load_config, model_cfg

PROMPT = ("A candid, photorealistic phone photo on a train platform of a fictional city at dusk: three commuters "
          "waiting, faces clearly visible, one checking a phone, and a large departures board on the right listing "
          "four trains. No real logos, brand names, transit agencies or real places anywhere.")
MIME = {b"\xff\xd8\xff": "jpeg", b"\x89PN": "png", b"RIF": "webp"}
BOX = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")


def boxes(text: str) -> list[dict]:
    """SAM 3.1's image records: `<0f>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|><|mask;...|>,...` (pixels)."""
    return [{"id": int(i), "x1": int(a), "y1": int(b), "x2": int(c), "y2": int(d), "w": int(w), "h": int(h),
             **({"c": float(conf)} if conf else {})} for i, a, b, c, d, w, h, conf in BOX.findall(text)]


async def segment(client: AsyncOpenAI, image_url: str, concept: str) -> tuple[list[dict], float]:
    t = time.perf_counter()
    text = ""
    stream = await client.responses.create(
        model="sam-3.1", stream=True, metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept}, {"type": "input_image", "image_url": image_url}]}])
    async for ev in stream:
        if ev.type == "response.output_text.delta":
            text += ev.delta
    return boxes(text), time.perf_counter() - t


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/privacy")
    ap.add_argument("--image", help="segment this image instead of generating one")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    m = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=180)
    report: dict = {}
    if args.image:
        png = Path(args.image).read_bytes()
    else:
        t = time.perf_counter()
        img = await client.images.generate(model="muse-image-1.0", prompt=PROMPT, n=1)
        report["muse_image_s"] = round(time.perf_counter() - t, 2)
        first = img.data[0]
        if getattr(first, "b64_json", None):
            png = base64.b64decode(first.b64_json)
        else:
            png = httpx.get(first.url, timeout=60).content
        report["muse_image_cost_usd"] = 0.01
    (out / "photo.png").write_bytes(png)
    data_url = f"data:image/{MIME.get(png[:3], 'png')};base64," + base64.b64encode(png).decode()
    faces, s_faces = await segment(client, data_url, "face")
    board, s_board = await segment(client, data_url, "departures board")
    report.update({"faces": faces, "sam_faces_s": round(s_faces, 2), "board": board, "sam_board_s": round(s_board, 2),
                   "sam_cost_usd": 2 * 0.0025, "bytes": len(png)})
    (out / "boxes.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("faces", "board")}), "faces:", len(faces),
          "boards:", len(board))


if __name__ == "__main__":
    asyncio.run(main())
