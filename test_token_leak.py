"""
Off-device regression tests for the last token of an answer leaking into the
next one (Ftelf/rkllama#19).

The RKLLM callback appends tokens to the module-level list global_text from the
model thread; run_rkllm_worker drains that list, then join()s the thread with a
short timeout. A token appended after the drain but before the thread ends was
never sent: the loop saw the thread finished and exited, the token stayed in
global_text, and the next inference sent it first (seen on RK3588 hardware:
' NameThe capital of France is Paris.').

The race is made deterministic with a fake thread whose join() runs the whole
model run, so every token is appended after the loop's first drain.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
import multiprocessing
import sys
import threading
import types

from test_worker_pipe_guard import _install_fakes, _load_worker_module


class _RunInsideJoinThread:
    """Stands in for threading.Thread in run_rkllm_worker: start() does nothing,
    join() runs the target, so the model 'appends its tokens' only after the
    worker's first drain and is finished when the worker next looks."""

    def __init__(self, target=None, args=(), kwargs=None, **_):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
        self._done = False

    def start(self):
        pass

    def join(self, timeout=None):
        if not self._done:
            self._done = True
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return not self._done


def _worker_with_late_tokens(monkeypatch, answers):
    """Worker module whose fake RKLLM.run appends the next answer's tokens and its
    metrics, one answer per call."""
    FakeRKLLM = _install_fakes()
    worker_mod = _load_worker_module()
    callback = sys.modules["rkllama.api.callback"]
    pending = list(answers)

    def run(self, *args):
        tokens = pending.pop(0)
        callback.global_text.extend(tokens)
        callback.global_metrics[:] = [3, len(tokens), 0.1, 0.2]

    monkeypatch.setattr(FakeRKLLM, "run", run)
    fake_threading = types.SimpleNamespace(**{name: getattr(threading, name) for name in dir(threading) if not name.startswith("__")})
    fake_threading.Thread = _RunInsideJoinThread
    monkeypatch.setattr(worker_mod, "threading", fake_threading)
    return worker_mod, callback


def _run_inferences(worker_mod, n_requests):
    """Run n inference requests, then unload; return the text each client received."""
    task_queue = multiprocessing.Queue()
    parents = []
    for _ in range(n_requests):
        parent, child = multiprocessing.Pipe()
        parents.append(parent)
        task_queue.put((child, worker_mod.WORKER_TASK_INFERENCE, "m", "t", ([1, 2, 3], None), {}))
    _, unload_child = multiprocessing.Pipe()
    task_queue.put((unload_child, worker_mod.WORKER_TASK_UNLOAD_MODEL, None, None, None, None))
    worker_mod.run_rkllm_worker("fake-model", task_queue, multiprocessing.Queue(), multiprocessing.Value("b", False),
                                model_path="unused", model_dir="unused")

    texts = []
    for parent in parents:
        received = []
        while parent.poll(2):
            message = parent.recv()
            if isinstance(message, tuple) and message and message[0] == worker_mod.WORKER_TASK_FINISHED:
                break
            assert worker_mod.worker_error_message(message) is None, message
            received.append(message)
        else:
            raise AssertionError(f"no WORKER_TASK_FINISHED after {received!r}")
        texts.append("".join(received))
    return texts


def test_tokens_appended_after_the_last_drain_reach_their_own_answer(monkeypatch):
    worker_mod, callback = _worker_with_late_tokens(monkeypatch, [["Hello", " Name"]])
    assert _run_inferences(worker_mod, 1) == ["Hello Name"]
    assert callback.global_text == [], "a token stayed behind for the next request"


def test_a_cut_answer_leaves_no_token_for_the_next_request(monkeypatch):
    worker_mod, _ = _worker_with_late_tokens(monkeypatch, [["Hello", " Name"], ["The capital", " is Paris."]])
    assert _run_inferences(worker_mod, 2) == ["Hello Name", "The capital is Paris."]


def test_tokens_left_from_an_earlier_run_are_not_sent(monkeypatch):
    """An aborted or failed run can leave tokens in global_text; the next answer
    must start with its own first token."""
    worker_mod, callback = _worker_with_late_tokens(monkeypatch, [["The capital", " is Paris."]])
    callback.global_text[:] = [" stale"]
    assert _run_inferences(worker_mod, 1) == ["The capital is Paris."]
