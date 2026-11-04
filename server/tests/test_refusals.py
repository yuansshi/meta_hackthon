"""A brain the API refused stays skipped across a restart (Fly's trial restarts the server every 5 minutes)."""

import time

from earshot import agent


def test_a_refused_brain_stays_skipped_across_a_restart(tmp_path, monkeypatch):
    path = tmp_path / "refused.json"
    monkeypatch.setattr(agent, "UNAVAILABLE", {"muse-spark": time.monotonic() + 600, "old": time.monotonic() - 1})
    agent.save_refusals(path)
    monkeypatch.setattr(agent, "UNAVAILABLE", {})  # the restart: nothing in memory
    agent.restore_refusals(path)
    assert 590 < agent.UNAVAILABLE["muse-spark"] - time.monotonic() <= 600
    assert "old" not in agent.UNAVAILABLE  # already expired before the restart


def test_a_brain_that_answers_again_is_forgotten(tmp_path, monkeypatch):
    path = tmp_path / "refused.json"
    monkeypatch.setattr(agent, "UNAVAILABLE", {"muse-spark": time.monotonic() + 600})
    agent.save_refusals(path)
    agent.UNAVAILABLE.pop("muse-spark")  # a check found it live
    agent.save_refusals(path)
    monkeypatch.setattr(agent, "UNAVAILABLE", {})
    agent.restore_refusals(path)
    assert agent.UNAVAILABLE == {}


def test_a_missing_stale_or_broken_file_restores_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "UNAVAILABLE", {})
    agent.restore_refusals(tmp_path / "missing.json")
    (tmp_path / "stale.json").write_text('{"muse-spark": 1}')  # a deadline long past
    agent.restore_refusals(tmp_path / "stale.json")
    (tmp_path / "broken.json").write_text("not json")
    agent.restore_refusals(tmp_path / "broken.json")
    (tmp_path / "odd.json").write_text('["muse-spark"]')
    agent.restore_refusals(tmp_path / "odd.json")
    assert agent.UNAVAILABLE == {}
