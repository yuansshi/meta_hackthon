import asyncio
import json

from fakes import CLOSE, FakeConnect, FakeWS

from earshot.config import load_config
from earshot.relay import AsrRelay


def fast_cfg(**over) -> dict:
    return {**load_config()["asr"], "frame_ms": 10, **over}  # 10 ms ticks keep the tests quick


async def noop(evt):
    pass


def test_handshake_sets_endpointing_explicitly():
    hs = AsrRelay(fast_cfg(), "k", noop).handshake()
    assert hs["mode"] == "ENDPOINTING"  # the API default is PUSH_TO_TALK
    assert hs["audioEncoding"] == "PCM_16KHZ"
    assert hs["authorization"]["accessToken"] == "Bearer k"
    assert "Earshot" in hs["keywords"]


def test_jitter_buffer_primes_drains_and_reprimes():
    r = AsrRelay(fast_cfg(prebuffer_frames=2), "k", noop)
    a, b = b"\1" * r.frame_bytes, b"\2" * r.frame_bytes
    r.feed(a)
    assert r.next_frame() == bytes(r.frame_bytes)  # not primed yet: silence
    r.feed(b)
    assert (r.next_frame(), r.next_frame()) == (a, b)
    r.feed(b"\3" * 10)
    assert r.next_frame() == (b"\3" * 10).ljust(r.frame_bytes, b"\0")  # underrun: pad, then re-prime
    assert r.underruns == 1 and not r.primed
    assert r.next_frame() == bytes(r.frame_bytes)


def test_backlog_is_capped():
    r = AsrRelay(fast_cfg(max_backlog_s=1.0), "k", noop)
    r.feed(bytes(16000 * 2 * 3))  # 3 s arrives at once after a cellular stall
    assert len(r.buf) == 16000 * 2
    assert abs(r.dropped_s - 2.0) < 1e-6


async def test_streams_silence_on_a_clock_and_forwards_events():
    events = []

    async def on_event(evt):
        events.append(evt)

    ws = FakeWS([{"sessionId": "s1"}, {"type": "speechComplete", "turnId": 1, "transcript": "hello"}])
    r = AsrRelay(fast_cfg(), "k", on_event, connect=FakeConnect([ws]))
    task = asyncio.create_task(r.run())
    await asyncio.sleep(0.15)
    assert json.loads(ws.sent[0])["model"] == "muse-voice-transcribe-1.0"
    frames = [f for f in ws.sent[1:] if isinstance(f, bytes)]
    assert len(frames) >= 8 and all(f == bytes(r.frame_bytes) for f in frames)  # silence keeps the stream alive
    assert events == [{"type": "speechComplete", "turnId": 1, "transcript": "hello"}]
    assert r.session_id == "s1"
    await r.close()
    task.cancel()


async def test_reconnects_after_the_server_closes():
    first = FakeWS([{"sessionId": "s1"}, CLOSE])
    second = FakeWS([{"sessionId": "s2"}])
    connect = FakeConnect([first, second])
    r = AsrRelay(fast_cfg(), "k", noop, connect=connect)
    task = asyncio.create_task(r.run())
    await asyncio.sleep(0.8)  # first backoff is 0.5 s
    assert r.sessions == 2 and r.session_id == "s2"
    assert r.closes[0][0] == 1000
    await r.close()
    task.cancel()


async def test_rejected_handshake_is_retried():
    bad = FakeWS([{"type": "error", "message": "unauthorized"}])
    good = FakeWS([{"sessionId": "s2"}])
    r = AsrRelay(fast_cfg(), "k", noop, connect=FakeConnect([bad, good]))
    task = asyncio.create_task(r.run())
    await asyncio.sleep(0.8)
    assert r.session_id == "s2"
    await r.close()
    task.cancel()
