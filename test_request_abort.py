"""
Off-device regression tests for issue #37: one request's timeout aborted the answer
of another request on the same model.

The abort was one shared flag per worker (``Worker.abort_flag``). A reader that timed
out set it, and the worker applied it to whatever ran at that moment, or to the next
inference: a request queued behind a running one aborted that run (seen on RK3588 by
rk-sbox, run 20261126T145159Z_rkllama_fda265a_api-e2e-chaos), and a timed out embedding
left it set, so the next inference answered with no tokens.

Fix under test: ``TrackedConnection.request_abort()`` sends the abort on the request's
own pipe, so it reaches only that request's task. A task that is still queued when its
reader aborts or leaves is skipped.

Uses the in-process worker harness of test_client_drop.py (fake runtime, no NPU, no
fork) and the REAL TrackedConnection.
"""
import logging
import multiprocessing
import queue
import threading
import time
import types

from test_client_drop import (  # noqa: F401  (env: fixture of the in-process worker harness)
    LONG_RUN, READ_TIMEOUT, SHORT_RUN, _assert_complete_answer, _inference, _read_answer, _recv,
    _start_worker, env)

EMBED_RUN = 100  # tokens of the fake embedding run: about 1 s


def _reader(env, conn, request_id=None):
    """The server's end of a request pipe, as WorkerManager._send_tracked_task returns it."""
    worker = types.SimpleNamespace(process=None, worker_model_info=types.SimpleNamespace(model="fake-model"))
    return env.worker_mod.TrackedConnection(conn, lambda: None, worker, request_id)


def _time_out(reader):
    """What every reader-timeout branch in server_utils.py does: abort, then close."""
    reader.request_abort()
    reader.close()


def _queue(env, task, n_tokens, tag, request_id):
    parent_conn, child_conn = multiprocessing.Pipe()
    env.tasks.put((child_conn, task, "mode", "type", (n_tokens, tag), {}, request_id))
    env.tasks.put((None, env.worker_mod.WORKER_TASK_CLEAR_CACHE, None, None, None, None))
    return _reader(env, parent_conn, request_id)


def _wait_for_runs(env, n):
    """Until the fake runtime has started n runs (it notifies only when a run ends)."""
    deadline = time.monotonic() + READ_TIMEOUT
    while len(env.runtime.runs) < n:
        assert time.monotonic() < deadline, env.runtime.runs
        time.sleep(0.001)


def _worker_lines(caplog, request_id):
    return [r.getMessage() for r in caplog.records
            if r.name == "rkllama.worker" and f"request {request_id} " in r.getMessage()]


def test_a_queued_request_that_times_out_does_not_abort_the_running_one(env, caplog):
    """Case 1 (rk-sbox F-018): B waits behind A on the same model and its reader times out."""
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    running = _reader(env, _inference(env, LONG_RUN, "a"))
    queued = _queue(env, env.worker_mod.WORKER_TASK_INFERENCE, SHORT_RUN, "b", "bbbb0002")
    _start_worker(env)
    assert _recv(running) == "a0,"
    _time_out(queued)
    assert env.runtime.active == 1, "A was not running when B timed out: the test proves nothing"
    answer = ["a0,"] + _read_answer(env, running)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, answer, "a", LONG_RUN)
    assert env.runtime.abort_calls == [], f"abort() was called: {env.runtime.abort_calls}"
    assert [run["tag"] for run in env.runtime.runs] == ["a"], "B's run started although its reader had left"
    assert _worker_lines(caplog, "bbbb0002") == ["request bbbb0002 run skipped outcome=aborted model=fake-model"]


def test_an_embedding_that_times_out_does_not_abort_the_next_inference(env):
    """Case 2: the embedding reader times out while the embedding runs; the inference behind it answers in full."""
    embedding = _queue(env, env.worker_mod.WORKER_TASK_EMBEDDING, EMBED_RUN, "e", "eeee0001")
    inference = _reader(env, _inference(env, SHORT_RUN, "n"))
    _start_worker(env)
    _wait_for_runs(env, 1)
    _time_out(embedding)
    assert env.runtime.active == 1, "the embedding was not running when it timed out: the test proves nothing"
    answer = _read_answer(env, inference)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, answer, "n", SHORT_RUN)
    assert env.runtime.abort_calls == [], f"abort() was called: {env.runtime.abort_calls}"


def test_an_abort_after_the_own_run_ended_does_not_abort_the_next_inference(env):
    """A reader that times out just as its run ends: the abort stays on its own pipe."""
    first = _reader(env, _inference(env, SHORT_RUN, "a"))
    # The worker waits for tasks: the next request comes only after the abort
    env.worker_thread = threading.Thread(
        target=env.worker_mod.run_rkllm_worker, args=("fake-model", env.tasks, queue.Queue()),
        kwargs={"model_path": "unused", "model_dir": "unused"}, daemon=True)
    env.worker_thread.start()
    first_answer = _read_answer(env, first)
    _time_out(first)
    second = _reader(env, _inference(env, SHORT_RUN, "b"))
    env.tasks.put((None, env.worker_mod.WORKER_TASK_UNLOAD_MODEL, None, None, None, None))
    second_answer = _read_answer(env, second)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, first_answer, "a", SHORT_RUN)
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)
    assert env.runtime.abort_calls == [], f"abort() was called: {env.runtime.abort_calls}"


def test_a_request_that_times_out_while_queued_is_skipped_and_the_next_one_runs(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    running = _reader(env, _inference(env, LONG_RUN, "a"))
    queued_inference = _queue(env, env.worker_mod.WORKER_TASK_INFERENCE, SHORT_RUN, "b", "bbbb0002")
    queued_embedding = _queue(env, env.worker_mod.WORKER_TASK_EMBEDDING, SHORT_RUN, "e", "eeee0003")
    live = _reader(env, _inference(env, SHORT_RUN, "c"))
    _start_worker(env)
    assert _recv(running) == "a0,"
    _time_out(queued_inference)
    _time_out(queued_embedding)
    running.request_abort()  # A is not under test here: end it early
    _read_answer(env, running)
    answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, answer, "c", SHORT_RUN)
    assert [run["tag"] for run in env.runtime.runs] == ["a", "c"], env.runtime.runs
    assert _worker_lines(caplog, "bbbb0002") == ["request bbbb0002 run skipped outcome=aborted model=fake-model"]
    assert _worker_lines(caplog, "eeee0003") == ["request eeee0003 run skipped outcome=aborted model=fake-model"]


def test_a_request_whose_client_left_while_queued_is_skipped(env, caplog):
    """A reader that closed its end without an abort (e.g. garbage collected) is not run either."""
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    running = _reader(env, _inference(env, LONG_RUN, "a"))
    left = _queue(env, env.worker_mod.WORKER_TASK_INFERENCE, SHORT_RUN, "b", "bbbb0002")
    _start_worker(env)
    assert _recv(running) == "a0,"
    left.close()
    running.request_abort()  # A is not under test here: end it early
    _read_answer(env, running)
    env.worker_thread.join(READ_TIMEOUT)

    assert [run["tag"] for run in env.runtime.runs] == ["a"], env.runtime.runs
    assert _worker_lines(caplog, "bbbb0002") == ["request bbbb0002 run skipped outcome=client gone model=fake-model"]


def test_a_request_that_times_out_during_its_own_run_is_still_aborted(env):
    """The #24 abort path is unchanged: the run is stopped, FINISHED with zero stats, the next request runs."""
    first = _reader(env, _inference(env, LONG_RUN, "a"))
    live = _reader(env, _inference(env, SHORT_RUN, "b"))
    _start_worker(env)
    assert _recv(first) == "a0,"
    first.request_abort()  # a reader that still reads the outcome
    first_answer = _read_answer(env, first)
    second_answer = _read_answer(env, live)
    env.worker_thread.join(READ_TIMEOUT)

    assert first_answer[-1] == (env.worker_mod.WORKER_TASK_FINISHED, 0, 0, 0, 0), first_answer
    assert env.runtime.runs[0]["aborted"] and env.runtime.runs[0]["produced"] < LONG_RUN, env.runtime.runs
    assert env.runtime.abort_calls == [1], env.runtime.abort_calls
    _assert_complete_answer(env, second_answer, "b", SHORT_RUN)


def test_request_abort_on_a_closed_pipe_does_not_raise(env):
    parent_conn, child_conn = multiprocessing.Pipe()
    reader = _reader(env, parent_conn)
    child_conn.close()
    parent_conn.close()
    reader.request_abort()
