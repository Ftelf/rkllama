"""
Off-device tests for issue #29, worker side: the request id in the worker and manager log.

- the worker loop in-process on a thread (test_client_drop.py): ``run started`` /
  ``run ended`` lines around a finished, dropped, aborted and failed run;
- the REAL WorkerManager with fake workers (test_dead_worker.py): the id on the task,
  the model lines of a request, the reason of every unload.
Log lines only: the existing lines keep their text.
"""
import logging
import multiprocessing
import sys
import types
from datetime import datetime, timedelta

import pytest

from test_client_drop import (  # noqa: F401  (env: fixture of the in-process worker harness)
    LONG_RUN, SHORT_RUN, _ConnFailingOnSecondSend, _drop_after_first_token, _read_answer, _recv, _start_worker, env)
from test_dead_worker import (
    _FakeProcess, _dead_fake_worker, _fake_worker, _live_fake_worker, _registering_loader,
    _real_worker, _started_process, _kill)
from test_dead_worker import env as manager_env  # noqa: F401  (fixture: a real WorkerManager)


# --- the worker: the bracket around each run ---------------------------------------------

def worker_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "rkllama.worker"]


def run_lines(caplog, request_id):
    return [line for line in worker_lines(caplog) if f"request {request_id} " in line]


def _task(env, n_tokens, tag, request_id, child_conn=None):
    """An INFERENCE task as WorkerManager._send_task_to queues it: the request id is the 7th element."""
    parent_conn = None
    if child_conn is None:
        parent_conn, child_conn = multiprocessing.Pipe()
    env.tasks.put((child_conn, env.worker_mod.WORKER_TASK_INFERENCE, "mode", "type", (n_tokens, tag), {}, request_id))
    env.tasks.put((None, env.worker_mod.WORKER_TASK_CLEAR_CACHE, None, None, None, None))
    return parent_conn


def test_worker_brackets_a_finished_run_with_the_request_id(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    conn = _task(env, SHORT_RUN, "a", "aaaa0001")
    _start_worker(env)
    _read_answer(env, conn)
    env.worker_thread.join(5)

    assert run_lines(caplog, "aaaa0001") == [
        "request aaaa0001 run started task=inference model=fake-model",
        f"request aaaa0001 run ended outcome=finished tokens={SHORT_RUN} model=fake-model",
    ]


def test_worker_logs_a_dropped_client_between_started_and_ended_of_its_request(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    dropped = _task(env, LONG_RUN, "a", "aaaa0001")
    live = _task(env, SHORT_RUN, "b", "bbbb0002")
    _start_worker(env)
    _drop_after_first_token(env, dropped)
    _read_answer(env, live)
    env.worker_thread.join(5)

    lines = worker_lines(caplog)
    started = lines.index("request aaaa0001 run started task=inference model=fake-model")
    ended = lines.index("request aaaa0001 run ended outcome=client gone model=fake-model")
    next_started = lines.index("request bbbb0002 run started task=inference model=fake-model")
    disconnected = next(i for i, line in enumerate(lines) if "disconnected during the answer" in line)
    stopped = next(i for i, line in enumerate(lines) if "stopped after" in line)
    # the path can be read from the log: started, the client left, the run stopped, ended, the next one starts
    assert started < disconnected < stopped < ended < next_started, lines
    assert run_lines(caplog, "bbbb0002")[-1].startswith("request bbbb0002 run ended outcome=finished"), lines


def test_worker_logs_an_aborted_run(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    first = _task(env, LONG_RUN, "a", "aaaa0001")
    _start_worker(env)
    assert _recv(first) == "a0,"
    first.send(env.worker_mod.WORKER_TASK_ABORT_REQUEST)
    _read_answer(env, first)
    env.worker_thread.join(5)

    assert run_lines(caplog, "aaaa0001")[-1] == "request aaaa0001 run ended outcome=aborted tokens=0 model=fake-model"


def test_worker_logs_a_run_that_failed_inside_the_loop(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    parent_conn, child_conn = multiprocessing.Pipe()
    _task(env, LONG_RUN, "a", "aaaa0001", child_conn=_ConnFailingOnSecondSend(child_conn))
    _start_worker(env)
    _read_answer(env, parent_conn)
    env.worker_thread.join(5)

    assert run_lines(caplog, "aaaa0001") == [
        "request aaaa0001 run started task=inference model=fake-model",
        "request aaaa0001 run ended outcome=error model=fake-model",
    ]


def test_a_task_without_an_id_adds_no_request_line(env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    parent_conn, child_conn = multiprocessing.Pipe()
    env.tasks.put((child_conn, env.worker_mod.WORKER_TASK_INFERENCE, "mode", "type", (SHORT_RUN, "a"), {}))
    env.tasks.put((None, env.worker_mod.WORKER_TASK_CLEAR_CACHE, None, None, None, None))
    _start_worker(env)
    _read_answer(env, parent_conn)
    env.worker_thread.join(5)

    assert [line for line in worker_lines(caplog) if line.startswith("request ")] == []
    assert "Running inference for model fake-model..." in worker_lines(caplog)


# --- the worker manager ------------------------------------------------------------------

def manager_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "rkllama.worker"]


def test_task_carries_the_request_id_as_its_last_element(manager_env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    worker = _live_fake_worker("m")
    manager_env.manager.workers["m"] = worker

    manager_env.manager._send_task_to(worker, "m", ("TASK", 1, 2, 3, 4), request_id="cafe0001")
    manager_env.manager._send_task_to(worker, "m", ("TASK", 1, 2, 3, 4))

    with_id, without_id = worker.task_queue.items
    assert with_id[1:] == ("TASK", 1, 2, 3, 4, "cafe0001")
    assert without_id[1:] == ("TASK", 1, 2, 3, 4), "a task without a request keeps its old shape (RKNN workers)"
    assert "request cafe0001 queued model=m" in manager_lines(caplog)
    assert len([line for line in manager_lines(caplog) if " queued " in line]) == 1


def test_load_and_begin_request_logs_reuse_load_and_reload(manager_env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    manager = manager_env.manager
    manager.set_model_loader(_registering_loader(manager_env, []))

    assert manager.load_and_begin_request("m", request_id="aaaa0001") is None          # not loaded: loads
    assert manager.load_and_begin_request("m", request_id="aaaa0002") is None          # loaded: reused
    manager.workers["m"] = _dead_fake_worker("m")
    assert manager.load_and_begin_request("m", request_id="aaaa0003") is None          # dead worker: reload

    lines = manager_lines(caplog)
    assert "request aaaa0001 loading (not loaded) model=m" in lines
    assert "request aaaa0002 model reused model=m" in lines
    assert "request aaaa0003 loading (reload after a dead worker) model=m" in lines
    assert "Unloading model m (reason: dead worker)" in lines
    assert sum(1 for line in lines if line.startswith("Unloading model m")) == 1


def test_load_without_a_request_id_adds_no_request_line(manager_env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    manager_env.manager.set_model_loader(_registering_loader(manager_env, []))
    manager_env.manager.load_and_begin_request("m")
    manager_env.manager.load_and_begin_request("m")
    assert [line for line in manager_lines(caplog) if line.startswith("request ")] == []


def test_dead_worker_seen_by_a_request_is_logged_with_its_id(manager_env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    process = _started_process()
    worker = _real_worker("m", process)
    try:
        manager_env.manager.workers["m"] = worker
        conn = manager_env.manager._send_tracked_task("m", ("TASK",), request_id="dead0001")
        assert conn.worker_gone is False
        _kill(process)
        assert conn.poll(5) is True
        assert conn.worker_gone is True
        conn.close()
    finally:
        worker.task_queue.close()
        if process.is_alive():
            process.kill()
        process.join(10)

    assert "request dead0001 worker exited exitcode=-9 model=m" in manager_lines(caplog)


def test_pipe_closed_without_a_result_is_worker_gone():
    worker_mod = sys.modules["rkllama.api.worker"]

    class ClosedPipe:
        def recv(self):
            raise EOFError

        def close(self):
            pass

    worker = types.SimpleNamespace(process=types.SimpleNamespace(), pipe_closed_without_result=False)
    conn = worker_mod.TrackedConnection(ClosedPipe(), lambda: None, worker, "dead0002")
    assert conn.worker_gone is False
    with pytest.raises(EOFError):
        conn.recv()
    assert conn.worker_gone is True


@pytest.mark.parametrize("call, reason", [
    (lambda m: m.stop_worker("m", reason="api"), "api"),
    (lambda m: m.stop_worker("m"), "unspecified"),
    (lambda m: m.stop_all(), "shutdown"),
    (lambda m: m.stop_all(reason="api"), "api"),
])
def test_unload_logs_its_reason_once(manager_env, caplog, monkeypatch, call, reason):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    monkeypatch.setattr(manager_env.worker_mod, "is_rkllm_model", lambda name: True)
    monkeypatch.setattr(manager_env.worker_mod, "is_gguf_model", lambda name: False)
    manager_env.manager.workers["m"] = _fake_worker("m", _FakeProcess(alive=False))

    call(manager_env.manager)

    assert [line for line in manager_lines(caplog) if line.startswith("Unloading model m")] == \
        [f"Unloading model m (reason: {reason})"]
    assert "Worker m stopped." in manager_lines(caplog), "the existing line must stay"
    assert manager_env.manager.workers == {}


def test_expired_model_is_unloaded_for_being_idle(manager_env, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    monkeypatch.setattr(manager_env.worker_mod, "is_rkllm_model", lambda name: True)
    monkeypatch.setattr(manager_env.worker_mod, "is_gguf_model", lambda name: False)
    worker = _fake_worker("m", _FakeProcess(alive=False))
    worker.worker_model_info.expires_at = datetime.now() - timedelta(minutes=1)
    manager_env.manager.workers["m"] = worker

    manager_env.manager.unload_expired_models()

    assert "Unloading model m (reason: idle)" in manager_lines(caplog)


def test_unload_of_a_model_that_is_not_loaded_logs_nothing(manager_env, caplog):
    caplog.set_level(logging.INFO, logger="rkllama.worker")
    manager_env.manager.stop_worker("gone", reason="api")
    assert [line for line in manager_lines(caplog) if "Unloading" in line] == []
