"""Rewind: the last few minutes of what the mic heard nearby, so "what did she just say?" has an answer.

Only utterances that weren't said to the agent are kept: in memory only, for `rewind_s` seconds, and only in
hands-free mode (push-to-talk sends silence between presses). Nothing is written to disk. Receipts record how
many lines were heard, never the words, and "remember that" goes through the memory tool like any other note.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from .tools import Tool, obj

DESCRIPTION = (
    "What the microphone heard in the last few minutes that wasn't said to you, oldest first, including other "
    "people talking nearby. Use it when the user asks what someone just said, or about something they just heard: "
    "a name, an address, a number, an announcement. Answer with the detail they want. It only covers the last few "
    "minutes; if they want something kept, save it with remember.")


class Heard:
    """A rolling transcript of the last `window_s` seconds, held in memory only."""

    def __init__(self, window_s: float, clock: Callable[[], float]):
        self.window_s = window_s
        self.clock = clock
        self.lines: deque[tuple[float, str]] = deque()

    def add(self, text: str) -> None:
        if text := text.strip():
            self.lines.append((self.clock(), text))
        self.prune()

    def prune(self) -> None:
        cutoff = self.clock() - self.window_s
        while self.lines and self.lines[0][0] < cutoff:
            self.lines.popleft()

    def clear(self) -> None:
        self.lines.clear()

    def recent(self, minutes: object = None) -> list[str]:
        """Lines from the last `minutes` (the whole window by default), each with how long ago it was heard."""
        self.prune()
        try:
            span = min(float(minutes) * 60, self.window_s) if minutes else self.window_s
        except (TypeError, ValueError):
            span = self.window_s
        now = self.clock()
        return [f"{round(now - t)} s ago: {text}" for t, text in self.lines if now - t <= span]


def rewind_tool(heard: Heard, mode: Callable[[], str]) -> Tool:
    async def rewind(minutes: object = None) -> dict:
        if lines := heard.recent(minutes):
            return {"heard": lines}
        if mode() == "ptt":
            return {"heard": [], "note": "in push-to-talk mode the mic only listens while the button is held, so "
                                         "nothing nearby was heard; Hands-free mode turns Rewind on"}
        return {"heard": [], "note": f"nothing was heard in the last {heard.window_s / 60:g} minutes"}

    def receipt(result: dict) -> dict:  # receipts are kept on disk; other people's words are not
        return {"lines": len(result.get("heard") or []), **({"note": result["note"]} if result.get("note") else {})}

    return Tool("rewind", DESCRIPTION,
                obj({"minutes": {"type": "number", "description": f"How far back to look, at most "
                                                                  f"{heard.window_s / 60:g} minutes. Leave it "
                                                                  f"out to look at all of it."}}),
                "server", run=rewind, receipt=receipt)
