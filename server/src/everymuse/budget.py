"""Process-wide daily spend estimate for the public demo; new turns stop once `daily_usd` is reached."""

from __future__ import annotations

import json
import time
from pathlib import Path


def token_cost(m: dict, inp: int, cached: int, out: int) -> float:
    """$ for one call from token counts and the per-1M prices in config/models.yaml."""
    return ((inp - cached) * m["price_in"] + cached * m.get("price_cached_in", m["price_in"])
            + out * m["price_out"]) / 1e6


class Budget:
    def __init__(self, daily_usd: float, path: Path | None = None):
        self.daily_usd = daily_usd
        self.path = path
        self.day = time.strftime("%Y-%m-%d")
        self.usd = 0.0
        if path and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("day") == self.day:
                self.usd = saved["usd"]

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.usd = today, 0.0

    def add(self, usd: float) -> None:
        self._roll()
        self.usd += usd
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"day": self.day, "usd": round(self.usd, 6)}))

    @property
    def spent(self) -> float:
        self._roll()
        return self.usd

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.daily_usd
