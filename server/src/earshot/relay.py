"""Meta realtime ASR (Muse Voice Transcribe) for one client connection, fed on our own 80 ms clock.

Meta needs continuous real-time PCM, silence included: it closes a stream that falls behind or runs
more than ~5 s ahead, and a phone on cellular delivers audio in bursts. So client audio goes into a
jitter buffer, and every tick sends exactly one frame: buffered audio, or zeros when there is none.
Past `max_backlog_s` the oldest audio is dropped. Any close (including the 60-minute session cap)
reconnects with backoff; the caller sees one continuous stream of events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

log = logging.getLogger(__name__)

OnEvent = Callable[[dict], Awaitable[None]]


class AsrRelay:
    def __init__(self, cfg: dict, api_key: str, on_event: OnEvent, connect=ws_connect):
        self.cfg = cfg
        self.api_key = api_key
        self.on_event = on_event  # must return quickly: it runs inside the receive loop
        self.connect = connect
        self.rate = cfg["sample_rate"]
        self.frame_bytes = int(self.rate * cfg["frame_ms"] / 1000) * 2
        self.prebuffer = self.frame_bytes * cfg.get("prebuffer_frames", 2)
        self.max_backlog = int(self.rate * cfg["max_backlog_s"]) * 2
        self.buf = bytearray()
        self.primed = False
        self.sent_s = 0.0  # audio seconds streamed to Meta (billing)
        self.dropped_s = 0.0
        self.underruns = 0
        self.sessions = 0
        self.stalls = 0  # sessions the watchdog closed because Meta stopped answering
        self.failures = 0  # attempts in a row that never opened a session (a refused key, no credit, an outage)
        self.last_heard = 0.0
        self.closes: list[tuple[int | None, str]] = []  # (code, reason) per ended session, for spike S6
        self.session_id: str | None = None
        self._ws = None
        self._stop = asyncio.Event()

    def feed(self, pcm: bytes) -> None:
        self.buf += pcm
        over = len(self.buf) - self.max_backlog
        if over > 0:
            over += over % 2
            del self.buf[:over]
            self.dropped_s += over / 2 / self.rate

    def handshake(self) -> dict:
        c = self.cfg
        return {
            "authorization": {"accessToken": f"Bearer {self.api_key}"},
            "audioEncoding": c["encoding"],
            "model": c["model"],
            "mode": c["mode"],
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,  # many a second: the heartbeat the stall watchdog listens for
            "languageBias": c["languages"],
            "keywords": c["keywords"],
        }

    def next_frame(self) -> bytes:
        """One frame of buffered audio, or silence. Re-primes after an underrun so speech isn't chopped."""
        if not self.primed and len(self.buf) >= self.prebuffer:
            self.primed = True
        if self.primed and len(self.buf) >= self.frame_bytes:
            frame = bytes(self.buf[: self.frame_bytes])
            del self.buf[: self.frame_bytes]
            return frame
        if self.primed:  # ran dry mid-stream: flush the remainder padded with zeros, then re-prime
            self.underruns += 1
            self.primed = False
            frame = bytes(self.buf).ljust(self.frame_bytes, b"\0")
            self.buf.clear()
            return frame
        return bytes(self.frame_bytes)

    async def run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 0.5
            except (OSError, ConnectionClosed, InvalidHandshake, TimeoutError, RuntimeError) as e:
                log.warning("asr session ended: %r", e)
                self.failures += 1
                if self.failures == 3:  # tell the user once, instead of a microphone that silently hears nothing
                    await self.on_event({"type": "relayDown", "error": f"{type(e).__name__}: {e}"[:200]})
            except Exception:  # anything unexpected must not leave the session deaf: log it and reconnect
                log.exception("asr session crashed")
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), backoff)
            except TimeoutError:
                pass
            backoff = min(backoff * 2, 8.0)

    async def _session(self) -> None:
        url = f"{self.cfg['url']}?sessionId=stream-{uuid.uuid4()}"
        async with self.connect(url, max_size=None, ping_interval=20, open_timeout=10) as ws:
            self._ws = ws
            await ws.send(json.dumps(self.handshake()))
            ack = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if "sessionId" not in ack:
                raise RuntimeError(f"ASR handshake rejected: {ack}")
            self.session_id = ack["sessionId"]
            self.sessions += 1
            if self.failures >= 3:
                await self.on_event({"type": "relayUp"})
            self.failures = 0
            log.info("asr session %s open", self.session_id)
            self.last_heard = time.monotonic()
            sender = asyncio.create_task(self._send_loop(ws))
            watchdog = asyncio.create_task(self._watch(ws))
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    self.last_heard = time.monotonic()
                    evt = json.loads(raw)
                    if evt.get("type") == "audioProgress":
                        continue  # proof the stream is alive, not news for the session
                    if evt.get("type") == "error":
                        log.warning("asr error: %s", evt.get("message"))
                    await self.on_event(evt)
            finally:
                sender.cancel()
                watchdog.cancel()
                self.closes.append((ws.close_code, ws.close_reason or ""))
                self._ws = None
                log.info("asr session %s closed: %s %s", self.session_id, ws.close_code, ws.close_reason)

    async def _watch(self, ws) -> None:
        """A stream can stay open yet stop answering (seen in a 70-minute soak). Meta sends audioProgress many times
        a second, so a quiet stream has stalled: close it, and run() reconnects."""
        stall = self.cfg.get("stall_s", 5.0)
        while True:
            await asyncio.sleep(min(1.0, stall / 3))
            if time.monotonic() - self.last_heard > stall:
                self.stalls += 1
                log.warning("asr session %s heard nothing for %.0f s; reconnecting", self.session_id, stall)
                await ws.close(code=4000, reason="stalled")
                return

    async def _send_loop(self, ws) -> None:
        tick = self.cfg["frame_ms"] / 1000
        next_t = time.monotonic()
        while True:
            await ws.send(self.next_frame())
            self.sent_s += tick
            next_t += tick
            delay = next_t - time.monotonic()
            if delay < -1.0:  # the event loop stalled; resync instead of bursting to catch up
                next_t = time.monotonic()
                delay = 0
            await asyncio.sleep(max(0.0, delay))

    async def close(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "endStream"}))
            await asyncio.wait_for(ws.wait_closed(), 2)
        except (ConnectionClosed, TimeoutError, OSError):
            pass
        finally:
            await ws.close()
