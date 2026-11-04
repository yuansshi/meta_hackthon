import pytest

from earshot.config import load_config
from earshot.session import Gate


@pytest.fixture
def gate() -> Gate:
    return Gate(load_config()["turns"], "wake")


@pytest.mark.parametrize("heard, rest", [
    ("Hey Earshot, what's the weather?", "what's the weather?"),
    ("hey ear shot what time is it", "what time is it"),
    ("Okay, air shot, remind me to call mom", "remind me to call mom"),
    ("Earshott set a timer", "set a timer"),
    ("Hi, Earshot. Where am I?", "Where am I?"),
])
def test_wake_word_variants(gate, heard, rest):
    d = gate.decide(heard, 0)
    assert (d.accept, d.text, d.reason) == (True, rest, "wake")


def test_wake_word_alone_asks_for_more(gate):
    assert gate.decide("Earshot.", 0).reason == "wake_only"


def test_no_wake_word_is_ignored_hands_free_but_not_push_to_talk(gate):
    assert gate.decide("what's the weather like", 0).reason == "no_wake"
    gate.mode = "ptt"
    assert gate.decide("what's the weather like", 0).reason == "ptt"


def test_hotshot_is_not_the_wake_word(gate):
    assert not gate.decide("that guy is a hotshot lawyer", 0).accept


def test_follow_up_window_after_a_reply(gate):
    gate.device_speaking(True, "It's sunny right now.", 10)
    gate.device_speaking(False, None, 11)
    assert gate.decide("and what about tomorrow", 13).reason == "follow_up"
    assert gate.decide("and what about tomorrow", 20).reason == "no_wake"


def test_echo_of_own_speech_is_dropped(gate):
    said = "It's sunny and sixty eight degrees right now."
    gate.device_speaking(True, said, 5)
    assert gate.decide("Hey Earshot it's sunny", 6).reason == "echo:speaking"  # even with the wake word
    gate.device_speaking(False, None, 7)
    assert gate.decide("it's sunny and sixty eight degrees", 7.3).reason == "echo:speaking"  # echo tail
    assert gate.decide("it's sunny and sixty eight degrees", 8.0).reason == "echo:match"
    assert gate.decide("great, thanks", 8.0).reason == "follow_up"


def test_device_that_never_reports_tts_end_does_not_deafen_the_gate(gate):
    gate.device_speaking(True, "hello", 0)
    assert gate.decide("Hey Earshot are you there", 10).reason == "echo:speaking"
    assert gate.decide("Hey Earshot are you there", 31).accept
