import asyncio
import json

from fakes import CLOSE, FakeConnect, FakeWS

from everymuse.config import load_config
from everymuse.relay import AsrRelay


def fast_cfg(**over) -> dict:
    return {**load_config()["asr"], "frame_ms": 10, **over}  # 10 ms ticks keep the tests quick


async def noop(evt):
    pass


def test_handshake_sets_endpointing_explicitly():
    hs = AsrRelay(fast_cfg(), "k", noop).handshake()
    assert hs["mode"] == "ENDPOINTING"  # the API default is PUSH_TO_TALK
    assert hs["audioEncoding"] == "PCM_16KHZ"
    assert hs["authorization"]["accessToken"] == "Bearer k"
    assert "EveryMuse" in hs["keywords"]
    assert hs["emitAudioProgress"] is True  # the stall watchdog's heartbeat


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


async def test_a_stalled_stream_is_closed_and_reopened():
    events = []

    async def on_event(evt):
        events.append(evt)

    first = FakeWS([{"sessionId": "s1"}, {"type": "audioProgress", "audioProcessedMs": 80}])  # then nothing at all
    second = FakeWS([{"sessionId": "s2"}])
    r = AsrRelay(fast_cfg(stall_s=0.3), "k", on_event, connect=FakeConnect([first, second]))
    task = asyncio.create_task(r.run())
    await asyncio.sleep(1.1)  # 0.3 s of quiet, then the 0.5 s reconnect backoff
    assert r.stalls >= 1 and r.session_id == "s2"
    assert first.close_code == 4000
    assert events == []  # the heartbeat isn't passed on to the session
    await r.close()
    task.cancel()


async def test_a_refused_speech_service_is_reported_once_and_then_its_return():
    events = []

    async def on_event(evt):
        events.append(evt)

    refused = [FakeWS([{"type": "error", "message": "insufficient credit"}]),
               FakeWS([{"sessionId": "s2"}, CLOSE]),  # accepted, then closed: 1008 "Billing verification failed"
               FakeWS([{"sessionId": "s3"}, CLOSE])]
    working = FakeWS([{"sessionId": "s4"}, {"type": "audioProgress", "audioProcessedMs": 80}])
    r = AsrRelay(fast_cfg(), "k", on_event, connect=FakeConnect([*refused, working]))
    task = asyncio.create_task(r.run())
    await asyncio.sleep(4.5)  # backoff 0.5 + 1 + 2 s between the refusals
    kinds = [e["type"] for e in events]
    assert kinds.count("relayDown") == 1 and kinds[-1] == "relayUp" and r.session_id == "s4"
    await r.close()
    task.cancel()
