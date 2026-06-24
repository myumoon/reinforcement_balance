import multiprocessing as mp
import signal

from games.survivors.survivors_env_factory import _ignore_sigint_in_subprocess


def test_ignore_sigint_does_not_change_main_process_handler(monkeypatch):
    calls = []
    monkeypatch.setattr(mp, "parent_process", lambda: None)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: calls.append((sig, handler)))

    _ignore_sigint_in_subprocess()

    assert calls == []


def test_ignore_sigint_changes_child_process_handler(monkeypatch):
    calls = []
    monkeypatch.setattr(mp, "parent_process", lambda: object())
    monkeypatch.setattr(signal, "signal", lambda sig, handler: calls.append((sig, handler)))

    _ignore_sigint_in_subprocess()

    assert calls == [(signal.SIGINT, signal.SIG_IGN)]
