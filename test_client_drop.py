"""
Off-device regression tests for issue #24: a client that disconnects in the
middle of a streamed answer broke the next request and killed the model worker.

run_rkllm_worker sent every token with a plain child_conn.send. Once the client
had closed its end, that send raised BrokenPipeError, the generic error handler
closed the pipe and the worker took the next task while the model thread (the
native rkllm_run) was still running: the CLEAR_CACHE queued after every
inference ran during that run, and the next INFERENCE started a second run on
the same runtime handle (on RK3588 hardware: HTTP 500 for the next request,
then the worker died with SIGSEGV).

Fix under test: a failed token send stops the run (abort, wait for the model
thread, clear the cache) before the next task; the abort-flag path and the
generic error handler wait for the model thread in the same way.

The fake runtime counts its active runs on one shared handle, so "two runs at
once" and "cache cleared during a run" are observable. Its run() adds one token
every TOKEN_INTERVAL s and, once abort() was called, returns ABORT_LATENCY s
later (a real runtime ends its current step first). What rkllm_abort does to an
idle handle is unknown, so the fake assumes the worst: an abort with no run
active makes the next run stop at once. Whether the real rkllm_abort is safe
during decoding, and which native call crashed, cannot be checked off-device.

The partial-UTF-8 test drives the REAL callback.py (loaded by file path) with
real ctypes RKLLMResult structures from the real classes.py.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no ctypes
runtime). The worker runs in-process on a thread; no fork.
"""
import ctypes
import importlib.util
import logging
import multiprocessing
import queue
import re
import sys
import threading
import time
import types

import pytest

from test_token_leak import _RunInsideJoinThread
from test_worker_pipe_guard import SRC, _install_fakes, _load_worker_module

TOKEN_INTERVAL = 0.01
ABORT_LATENCY = 0.1
LONG_RUN = 200   # tokens: about 2 s, far longer than any test needs
SHORT_RUN = 5
READ_TIMEOUT = 5


class _Runtime:
    """Bookkeeping of the one runtime handle all runs of a worker share."""

    def __init__(self):
        self.cond = threading.Condition()
        self.active = 0
        self.max_active = 0
        self.runs = []               # one dict per run: tag, produced, aborted
        self.abort_calls = []        # active runs when abort() was called
        self.clear_cache_calls = []  # active runs when clear_cache() was called
        self.clear_cache_failures = 0  # the next n clear_cache() calls raise
        self.instances = []

    def wait_idle(self, timeout):
        with self.cond:
            return self.cond.wait_for(lambda: self.active == 0, timeout)


def _fake_rkllm_class(runtime, callback):
    class FakeRKLLM:
        def __init__(self, *args, **kwargs):
            self._abort = threading.Event()
            self._idle_abort = False  # sticky: abort() while no run was active
            runtime.instances.append(self)

        def run(self, inference_mode, model_input_type, model_input, options):
            n_tokens, tag = model_input
            record = {"tag": tag, "produced": 0, "aborted": False}
            self._abort.clear()
            with runtime.cond:
                runtime.active += 1
                runtime.max_active = max(runtime.max_active, runtime.active)
                runtime.runs.append(record)
                stopped_by_idle_abort, self._idle_abort = self._idle_abort, False
            try:
                if stopped_by_idle_abort:
                    record["aborted"] = True
                    return
                for k in range(n_tokens):
                    # Paces the tokens; True as soon as abort() was called
                    if self._abort.wait(TOKEN_INTERVAL):
                        record["aborted"] = True
                        # The runtime ends its current step before rkllm_run returns
                        # (models latency, not test synchronisation)
                        time.sleep(ABORT_LATENCY)
                        return
                    callback.global_text.append(f"{tag}{k},")
                    record["produced"] += 1
                callback.global_metrics[:] = [3, n_tokens, 0.1, 0.2]
            finally:
                with runtime.cond:
                    runtime.active -= 1
                    runtime.cond.notify_all()

        def abort(self):
            with runtime.cond:
                runtime.abort_calls.append(runtime.active)
                if runtime.active == 0:
                    self._idle_abort = True
                else:
                    self._abort.set()

        def clear_cache(self):
            with runtime.cond:
                runtime.clear_cache_calls.append(runtime.active)
                if runtime.clear_cache_failures > 0:
                    runtime.clear_cache_failures -= 1
                    raise RuntimeError("clear_cache failed")

        def release(self):
            pass

    return FakeRKLLM


@pytest.fixture
def env(monkeypatch):
    _install_fakes()
    worker_mod = _load_worker_module()
    callback = sys.modules["rkllama.api.callback"]
    runtime = _Runtime()
    monkeypatch.setattr(sys.modules["rkllama.api.rkllm"], "RKLLM", _fake_rkllm_class(runtime, callback))
    env = types.SimpleNamespace(worker_mod=worker_mod, runtime=runtime, tasks=queue.Queue(),
                                abort_flag=multiprocessing.Value("b", False), worker_thread=None)
    yield env
    # A run the worker left behind must not outlive the test
    for instance in runtime.instances:
        instance._abort.set()
    assert runtime.wait_idle(READ_TIMEOUT), "a fake model run did not end"
    if env.worker_thread is not None:
        env.worker_thread.join(READ_TIMEOUT)
        assert not env.worker_thread.is_alive(), "the worker did not exit after the unload task"


def _inference(env, n_tokens, tag, child_conn=None):
    """Queue an INFERENCE task the way WorkerManager.inference does (followed by
    its CLEAR_CACHE); return the client's end of the request pipe."""
    parent_conn = None
    if child_conn is None:
        parent_conn, child_conn = multiprocessing.Pipe()
    env.tasks.put((child_conn, env.worker_mod.WORKER_TASK_INFERENCE, "mode", "type", (n_tokens, tag), {}))
    env.tasks.put((None, env.worker_mod.WORKER_TASK_CLEAR_CACHE, None, None, None, None))
    return parent_conn


def _start_worker(env):
    env.tasks.put((None, env.worker_mod.WORKER_TASK_UNLOAD_MODEL, None, None, None, None))
    env.worker_thread = threading.Thread(
        target=env.worker_mod.run_rkllm_worker,
        args=("fake-model", env.tasks, queue.Queue(), env.abort_flag),
        kwargs={"model_path": "unused", "model_dir": "unused"}, daemon=True)
    env.worker_thread.start()


def _recv(conn):
    assert conn.poll(READ_TIMEOUT), f"no message from the worker within {READ_TIMEOUT}s"
    return conn.recv()


def _read_answer(env, conn):
    """Every message of one answer up to and including its final one."""
    worker_mod = env.worker_mod
    messages = []
    while True:
        message = _recv(conn)
        messages.append(message)
        finished = isinstance(message, tuple) and message and message[0] == worker_mod.WORKER_TASK_FINISHED
        if finished or worker_mod.worker_error_message(message) is not None:
            return messages


def _own_tokens(tag, n_tokens):
    return [f"{tag}{k}," for k in range(n_tokens)]


def _assert_complete_answer(env, messages, tag, n_tokens):
    """Exactly this run's tokens, then FINISHED with this run's stats."""
    assert messages[:-1] == _own_tokens(tag, n_tokens), messages
    assert messages[-1][:3] == (env.worker_mod.WORKER_TASK_FINISHED, 3, n_tokens), messages[-1]


def _drop_after_first_token(env, conn):
    """The client reads one token and closes its end (a "stop" click)."""
    first = _recv(conn)
    assert first == "a0,", first
    conn.close()


def _assert_one_stop_line(caplog, aborted):
    """The stop sequence logs one INFO line with its durations (hardware evidence of
    how long rkllm_abort takes) that says whether abort() was called."""
    lines = [r.getMessage() for r in caplog.records
             if r.name == "rkllama.worker" and r.levelno == logging.INFO and "stopped after" in r.getMessage()]
    assert len(lines) == 1, lines
    pattern = rf"Run of model fake-model stopped after \d+ ms \(aborted={aborted}, abort\+join \d+ ms\)"
    assert re.fullmatch(pattern, lines[0]), lines[0]


def _run_dropped_client_then_live_client(env):
    dropped = _inference(env, LONG_RUN, "a")
    live = _inference(env, SHORT_RUN, "b")
    _start_worker(env)
    _drop_after_first_token(env, dropped)
    return _read_answer(env, live)


def test_dropped_client_stops_the_run_before_the_next_task(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    second_answer = _run_dropped_client_then_live_client(env)

    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    runtime = env.runtime
    assert runtime.max_active == 1, f"{runtime.max_active} runs at once on one runtime handle"
    dropped_run = runtime.runs[0]
    assert dropped_run["tag"] == "a" and dropped_run["aborted"], runtime.runs
    assert dropped_run["produced"] < LONG_RUN, "the dropped run went on to its end"
    assert runtime.abort_calls == [1], runtime.abort_calls
    _assert_one_stop_line(caplog, aborted=True)
    # Nothing is reported to the closed pipe, and the drop is not handled as a task failure
    problems = [r.getMessage() for r in caplog.records if r.name == "rkllama.worker" and r.levelno >= logging.WARNING]
    assert problems == [], problems


def test_clear_cache_never_runs_during_a_run(env):
    _run_dropped_client_then_live_client(env)
    env.worker_thread.join(READ_TIMEOUT)

    calls = env.runtime.clear_cache_calls
    # the stop sequence of the dropped run + the CLEAR_CACHE after each inference
    assert len(calls) == 3, calls
    assert calls == [0, 0, 0], f"clear_cache ran during a run: active runs per call {calls}"


def test_abort_flag_path_waits_for_the_run_to_end(env):
    first = _inference(env, LONG_RUN, "a")
    live = _inference(env, SHORT_RUN, "b")
    _start_worker(env)
    # A reader that gives up on the answer but still reads the outcome
    assert _recv(first) == "a0,"
    env.abort_flag.value = True
    first_answer = _read_answer(env, first)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    # What the abort-flag path reports is unchanged: FINISHED with zero stats
    assert first_answer[-1] == (env.worker_mod.WORKER_TASK_FINISHED, 0, 0, 0, 0), first_answer
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    runtime = env.runtime
    assert runtime.abort_calls == [1], runtime.abort_calls
    assert runtime.max_active == 1, f"{runtime.max_active} runs at once on one runtime handle"
    assert runtime.clear_cache_calls == [0, 0, 0], f"clear_cache ran during a run: {runtime.clear_cache_calls}"
    assert not env.abort_flag.value, "the abort flag stayed set for the next request"


class _ConnFailingOnSecondSend:
    """A request pipe whose second send fails with an error that is not a
    closed pipe (anything unexpected inside the run loop); later sends work."""

    def __init__(self, conn):
        self._conn = conn
        self.sends = 0

    def send(self, value):
        self.sends += 1
        if self.sends == 2:
            raise RuntimeError("unexpected failure while sending a token")
        self._conn.send(value)

    def close(self):
        self._conn.close()


def test_exception_in_the_run_loop_does_not_leave_a_run_behind(env):
    parent_conn, child_conn = multiprocessing.Pipe()
    _inference(env, LONG_RUN, "a", child_conn=_ConnFailingOnSecondSend(child_conn))
    live = _inference(env, SHORT_RUN, "b")
    _start_worker(env)
    first_answer = _read_answer(env, parent_conn)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    # The client still gets the error report
    assert first_answer == ["a0,", env.worker_mod.WORKER_TASK_ERROR], first_answer
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    runtime = env.runtime
    assert runtime.runs[0]["aborted"], runtime.runs
    assert runtime.abort_calls == [1], runtime.abort_calls
    assert runtime.max_active == 1, f"{runtime.max_active} runs at once on one runtime handle"
    assert runtime.clear_cache_calls == [0, 0, 0], f"clear_cache ran during a run: {runtime.clear_cache_calls}"


def test_finished_requests_are_not_stopped(env):
    """Control: answers that are read to their end never go through the stop sequence."""
    first = _inference(env, SHORT_RUN, "a")
    second = _inference(env, SHORT_RUN, "b")
    _start_worker(env)
    first_answer = _read_answer(env, first)
    second_answer = _read_answer(env, second)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, first_answer, "a", SHORT_RUN)
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    runtime = env.runtime
    assert runtime.abort_calls == [], runtime.abort_calls
    # only the CLEAR_CACHE queued after each inference
    assert runtime.clear_cache_calls == [0, 0], runtime.clear_cache_calls
    assert [r["aborted"] for r in runtime.runs] == [False, False], runtime.runs


def test_client_gone_at_the_final_drain_stops_without_aborting_an_ended_run(env, caplog, monkeypatch):
    """The run has ended when the worker first finds the pipe closed (the final
    drain): the stop sequence must still run, but without abort() on the idle handle."""
    fake_threading = types.SimpleNamespace(**{n: getattr(threading, n) for n in dir(threading) if not n.startswith("__")})
    fake_threading.Thread = _RunInsideJoinThread  # the whole run happens inside the worker's join()
    monkeypatch.setattr(env.worker_mod, "threading", fake_threading)
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    dropped = _inference(env, SHORT_RUN, "a")
    live = _inference(env, SHORT_RUN, "b")
    dropped.close()  # gone before the worker sends anything: every token goes out in the final drain
    _start_worker(env)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    runtime = env.runtime
    assert runtime.abort_calls == [], f"abort() on an ended run: {runtime.abort_calls}"
    _assert_one_stop_line(caplog, aborted=False)
    assert runtime.max_active == 1, runtime.max_active
    # the stop sequence's clear_cache + the CLEAR_CACHE after each inference
    assert runtime.clear_cache_calls == [0, 0, 0], runtime.clear_cache_calls
    problems = [r.getMessage() for r in caplog.records if r.name == "rkllama.worker" and r.levelno >= logging.WARNING]
    assert problems == [], problems


def test_a_failing_stop_sequence_does_not_leave_the_abort_flag_set(env, caplog):
    env.runtime.clear_cache_failures = 1  # the stop sequence's clear_cache raises
    first = _inference(env, LONG_RUN, "a")
    live = _inference(env, SHORT_RUN, "b")
    _start_worker(env)
    assert _recv(first) == "a0,"
    env.abort_flag.value = True
    first_answer = _read_answer(env, first)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    # The failure is reported to its client and logged, not hidden
    assert first_answer[-1] == env.worker_mod.WORKER_TASK_ERROR, first_answer
    assert any("Failed executing task" in r.getMessage() for r in caplog.records if r.name == "rkllama.worker")
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    assert not env.abort_flag.value, "the abort flag stayed set for the next request"
    assert env.runtime.max_active == 1, env.runtime.max_active


# --- partial UTF-8 bytes, through the real callback.py -------------------------

def _load_real_callback(monkeypatch):
    """callback.py as the worker imports it. Its package-relative imports: the
    real classes.py (registered by _install_fakes) and an empty variables
    module (the real one starts a WorkerManager); numpy is only used for
    embeddings, so an empty stand-in is enough when it is not installed."""
    monkeypatch.setitem(sys.modules, "rkllama.api.variables", types.ModuleType("rkllama.api.variables"))
    if importlib.util.find_spec("numpy") is None:
        monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    spec = importlib.util.spec_from_file_location("rkllama.api.callback", str(SRC / "rkllama" / "api" / "callback.py"))
    callback_mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rkllama.api.callback", callback_mod)
    spec.loader.exec_module(callback_mod)
    return callback_mod


def _callback_driven_rkllm_class(callback_mod, classes, partial_sent):
    """A model whose run() reports its tokens through the real callback_impl,
    as librkllmrt does: model_input = (token byte strings, wait for abort)."""
    state = classes.LLMCallState

    def report(status, text=None, generate_tokens=0):
        result = classes.RKLLMResult()
        result.text = text
        result.perf.prefill_tokens = 3
        result.perf.generate_tokens = generate_tokens
        callback_mod.callback_impl(ctypes.pointer(result), None, status)

    class CallbackDrivenRKLLM:
        def __init__(self, *args, **kwargs):
            self._abort = threading.Event()

        def run(self, inference_mode, model_input_type, model_input, options):
            tokens, wait_for_abort = model_input
            self._abort.clear()
            for token in tokens:
                report(state.RKLLM_RUN_NORMAL, token)
            if wait_for_abort:
                partial_sent.set()
                self._abort.wait(READ_TIMEOUT)
                return
            report(state.RKLLM_RUN_FINISH, generate_tokens=len(tokens))

        def abort(self):
            self._abort.set()

        def clear_cache(self):
            pass

        def release(self):
            pass

    return CallbackDrivenRKLLM


@pytest.mark.parametrize("first_run_ends", ["aborted", "cut at the token limit"])
def test_partial_utf8_bytes_of_a_run_do_not_empty_the_next_answer(env, monkeypatch, first_run_ends):
    """A run that ends inside a multi-byte character (an abort, or max_new_tokens)
    leaves its first bytes in callback.split_byte_data; the next run must not
    start with them, or none of its tokens ever decodes."""
    callback_mod = _load_real_callback(monkeypatch)
    partial_sent = threading.Event()
    monkeypatch.setattr(sys.modules["rkllama.api.rkllm"], "RKLLM",
                        _callback_driven_rkllm_class(callback_mod, sys.modules["rkllama.api.classes"], partial_sent))
    aborted = first_run_ends == "aborted"
    # b"\xe4" is the first of the three bytes of U+4E2D
    first = _inference(env, [b"Hi ", b"\xe4"], aborted)
    live = _inference(env, [b"Hello", b" world"], False)
    _start_worker(env)
    first_answer = []
    if aborted:
        first_answer.append(_recv(first))
        assert partial_sent.wait(READ_TIMEOUT)
        env.abort_flag.value = True
    first_answer += _read_answer(env, first)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    assert first_answer[0] == "Hi ", first_answer
    assert callback_mod.split_byte_data == b"", "a later run kept the partial bytes"
    assert second_answer[:-1] == ["Hello", " world"], second_answer
    assert second_answer[-1][:3] == (env.worker_mod.WORKER_TASK_FINISHED, 3, 2), second_answer[-1]
