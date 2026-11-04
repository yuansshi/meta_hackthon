"""Find it: SAM 3.1 locates something in the photo the user asked about, so EveryMuse can say where it is.

Muse Spark reads the photo. When the question is "where is it?" or "show me", it calls `find_in_photo` with a short
noun phrase; SAM 3.1 segments that concept and returns boxes in pixels. This turns the boxes into spoken positions
("top right") and sends them to the client, which outlines them on the photo.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable

from .tools import Tool, obj

DESCRIPTION = (
    "Find where something is in the photo the user shared in the last two minutes, with Meta's SAM 3.1 segmentation. "
    "Use it when they ask where something is, or to show or point at something. Pass a short noun phrase naming a "
    "kind of object, like 'exit sign', 'departures board' or 'red backpack'. SAM finds objects, not text: for a line "
    "written on a sign, board or screen, pass the sign, board or screen itself. It returns where each match is, and "
    "the app outlines them on the photo. Say where in a few words.")
BOX = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")


def boxes(text: str) -> list[dict]:
    """SAM 3.1 image records (`<0f>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H;c=..|><|mask;...|>,...`), as fractions."""
    out = []
    for _, x1, y1, x2, y2, w, h, c in BOX.findall(text):
        w, h = int(w) or 1, int(h) or 1
        out.append({"x1": int(x1) / w, "y1": int(y1) / h, "x2": (int(x2) + 1) / w, "y2": (int(y2) + 1) / h,
                    "c": float(c) if c else None})
    return out


def where(b: dict) -> str:
    """A box as a listener would want it: 'top right', 'on the left', 'in the middle'."""
    cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    if (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) > 0.5:
        return "filling most of the photo"
    if row == "middle":
        return "in the middle" if col == "center" else f"on the {col}"
    return f"at the {row}" if col == "center" else f"{row} {col}"


async def locate(client, model: str, image_url: str, thing: str) -> tuple[list[dict], float]:
    t, text = time.perf_counter(), ""
    stream = await client.responses.create(
        model=model, stream=True, metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": thing}, {"type": "input_image", "image_url": image_url}]}])
    async for ev in stream:
        if ev.type == "response.output_text.delta":
            text += ev.delta
    return boxes(text), time.perf_counter() - t


def find_tool(cfg: dict, photo: Callable[[], str | None], client: Callable[[], object],
              highlight: Callable[[list[dict]], Awaitable[None]], spend: Callable[[float], None]) -> Tool:
    async def find_in_photo(thing: str) -> dict:
        image = photo()
        if not image:
            return {"error": "there's no photo from the last two minutes to look in"}
        thing = " ".join(str(thing).split())[:60] or "object"
        found, seconds = await locate(client(), cfg["model"], image, thing)
        spend(cfg["price_per_image"])
        if not found:
            return {"found": 0, "note": f"SAM 3.1 found no {thing} in the photo"}
        found.sort(key=lambda b: -(b["c"] or 0))
        await highlight(found[:5])
        return {"found": len(found), "where": [where(b) for b in found[:5]], "seconds": round(seconds, 1)}

    return Tool("find_in_photo", DESCRIPTION,
                obj({"thing": {"type": "string", "description": "A short noun phrase, like 'exit sign'."}}, ("thing",)),
                "server", run=find_in_photo)
