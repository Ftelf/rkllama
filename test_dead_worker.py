"""
Off-device regression tests for issue #23: a request to a model whose worker
process died (SIGKILL, OOM kill, segfault) waited the whole
max_seconds_waiting_worker_response and then failed with HTTP 500 KeyError.

Why it waited: the request pipe's worker end travels through the worker's task
queue. Pickling it there makes multiprocessing keep a duplicate of that end in
the parent until a process unpickles it, and a dead worker never does. So the
reader's end never sees EOF and poll(timeout) runs out. Then the reader looked
the worker up by model name to set its abort flag, but the monitor had removed
the dead entry meanwhile: KeyError.

Fix under test: the request pipe (TrackedConnection) also watches the worker
process, a request to a model with a dead worker loads it again, the reaper
never removes a replacement worker, and aborts go to the worker the request
was sent to.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no ctypes
runtime). Real processes use the spawn context, so no fork happens in this
multi-threaded test process.
"""
import multiprocessing
import signal
import threading
import time
import types

import pytest

from test_model_loading import _limit_models, _use_fake_workers
from test_worker_pipe_guard import _install_fakes, _load_worker_module

SPAWN = multiprocessing.get_context("spawn")


@pytest.fixture
def env(monkeypatch):
    """Fresh fakes + worker module + a WorkerManager built by the REAL __init__,
    minus its host side effects (no orphan killing, no monitor thread), as in
    test_model_loading.py. Restores signal handlers and empties the manager
    afterwards so its atexit stop_all() has nothing to stop."""
    _install_fakes()
    worker_mod = _load_worker_module()
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    monkeypatch.setattr(worker_mod, "_kill_orphaned_workers", lambda: None)
    monkeypatch.setattr(worker_mod.WorkerManager, "start_models_monitor", lambda self, interval=60: None)
    manager = worker_mod.WorkerManager()
    yield types.SimpleNamespace(worker_mod=worker_mod, manager=manager)
    manager.workers.clear()
    for sig, handler in saved.items():
        signal.signal(sig, handler)


# --- helpers ------------------------------------------------------------------

def _started_process():
    """A real worker-like child process (it only sleeps)."""
    process = SPAWN.Process(target=time.sleep, args=(60,), daemon=True)
    process.start()
    return process


def _kill(process):
    """SIGKILL the child the way the OOM killer does and wait until it is gone."""
    process.kill()
    process.join(10)
    assert process.exitcode == -signal.SIGKILL, f"child did not die from SIGKILL: exitcode={process.exitcode}"


def _real_worker(name, process):
    """A worker entry with a real process, a real task queue and a real abort flag."""
    return types.SimpleNamespace(
        process=process, task_queue=SPAWN.Queue(), abort_flag=SPAWN.Value('b', False),
        worker_model_info=types.SimpleNamespace(model=name, last_call=None, expires_at=None))


class _FakeProcess:
    """Stands in for a worker's multiprocessing.Process in the load/reap tests."""

    def __init__(self, alive, exitcode=None, on_is_alive=None):
        self.alive = alive
        self.exitcode = exitcode
        self.on_is_alive = on_is_alive
        self.joined = False

    def is_alive(self):
        if self.on_is_alive is not None:
            on_is_alive, self.on_is_alive = self.on_is_alive, None
            on_is_alive()
        return self.alive

    def join(self, timeout=None):
        self.joined = True


class _FakeQueue:
    def __init__(self):
        self.items = []
        self.closed = False

    def put(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True


def _fake_worker(name, process, domain=1):
    return types.SimpleNamespace(
        process=process, task_queue=_FakeQueue(), abort_flag=types.SimpleNamespace(value=False),
        worker_model_info=types.SimpleNamespace(model=name, size=0, base_domain_id=[domain],
                                                last_call=None, expires_at=None))


def _dead_fake_worker(name):
    return _fake_worker(name, _FakeProcess(alive=False, exitcode=-signal.SIGKILL))


def _live_fake_worker(name):
    return _fake_worker(name, _FakeProcess(alive=True))


# --- root cause: positive control -----------------------------------------------

def test_pipe_queued_to_a_dead_worker_never_reaches_eof():
    """POSITIVE CONTROL for the root cause: once the worker end of a pipe went
    through a multiprocessing Queue, closing it here does not give the reader
    EOF, because the queue keeps a duplicate until a process unpickles it. If
    this ever fails, the stdlib changed and the reasoning behind the
    TrackedConnection sentinel watch must be revisited."""
    task_queue = SPAWN.Queue()
    reader, worker_end = SPAWN.Pipe()
    try:
        task_queue.put((worker_end, "TASK"))
        task_queue.close()
        task_queue.join_thread()  # the feeder thread has pickled (and duplicated) the worker end
        worker_end.close()

        assert reader.poll(1.0) is False, "the reader got EOF: the queued worker end is not kept open any more"
    finally:
        reader.close()


# --- the reader of a request to a dead worker ------------------------------------

def test_reader_fails_fast_when_the_worker_died_with_the_task_queued(env):
    process = _started_process()
    worker = _real_worker("m", process)
    try:
        _kill(process)
        env.manager.workers["m"] = worker

        conn = env.manager._send_tracked_task("m", ("TASK",))
        started = time.monotonic()
        ready = conn.poll(5)
        waited = time.monotonic() - started

        assert ready is True, "poll() did not report the dead worker"
        assert waited < 1, f"the reader waited {waited:.1f}s for a worker that is already dead"
        with pytest.raises(EOFError, match="exitcode=-9"):
            conn.recv()
        conn.close()
        assert not env.manager.is_busy("m")
    finally:
        worker.task_queue.close()
        if process.is_alive():
            process.kill()
        process.join(10)


DIED_WITHOUT_FINISHING = "without finishing the request"


def _dead_worker_errors(caplog):
    return [r for r in caplog.records
            if r.name == "rkllama.worker" and r.levelname == "ERROR" and DIED_WITHOUT_FINISHING in r.getMessage()]


def test_answer_sent_before_the_worker_died_wins_over_its_death(env, caplog):
    """The worker sends its last message and then dies, so poll() finds the
    data and the dead process ready together. The data must win: the request
    is not reported as failed while its answer is still unread (poll checks
    the pipe before the sentinel)."""
    caplog.set_level("ERROR", logger="rkllama.worker")
    process = _started_process()
    worker = _real_worker("m", process)
    worker_end = None
    try:
        env.manager.workers["m"] = worker
        conn = env.manager._send_tracked_task("m", ("TASK",))

        # Play the worker: take the task (unpickles the worker end), answer, die
        worker_end, task = worker.task_queue.get(timeout=5)
        assert task == "TASK"
        worker_end.send("last token")
        _kill(process)

        assert conn.poll(5) is True
        assert _dead_worker_errors(caplog) == [], "the worker was reported dead while its answer was still unread"
        assert conn.recv() == "last token"

        # Positive control of the log check: once the answer is read, the death is reported
        started = time.monotonic()
        assert conn.poll(5) is True, "poll() did not report the dead worker after its last message"
        assert time.monotonic() - started < 1
        assert len(_dead_worker_errors(caplog)) == 1
        with pytest.raises(EOFError):
            conn.recv()
        conn.close()
    finally:
        if worker_end is not None:
            worker_end.close()
        worker.task_queue.close()
        if process.is_alive():
            process.kill()
        process.join(10)


def test_data_still_in_the_pipe_is_read_after_the_death_was_seen(env):
    """Once poll() has seen the dead worker, recv() still returns what is in
    the pipe before it raises EOFError (recv checks the pipe before raising).
    The test writes into the pipe after the death was seen, so the guard is
    hit on its own; with a real worker, all its writes happen before it exits,
    so this ordering is apparently rare."""
    process = _started_process()
    worker = _real_worker("m", process)
    worker_end = None
    try:
        env.manager.workers["m"] = worker
        conn = env.manager._send_tracked_task("m", ("TASK",))
        worker_end, task = worker.task_queue.get(timeout=5)
        _kill(process)

        assert conn.poll(5) is True, "poll() did not report the dead worker"
        worker_end.send("still buffered")

        assert conn.recv() == "still buffered", "recv() dropped data that was still in the pipe"
        with pytest.raises(EOFError, match="exitcode=-9"):
            conn.recv()
        conn.close()
    finally:
        if worker_end is not None:
            worker_end.close()
        worker.task_queue.close()
        if process.is_alive():
            process.kill()
        process.join(10)


def test_worker_without_sentinel_falls_back_to_plain_poll(env):
    """CONTROL: a worker whose process has no sentinel (here process None)
    keeps the plain pipe behaviour. Only _send_tracked_task builds a
    TrackedConnection, and GGUF Popen workers never go through it (their
    requests are proxied over HTTP); this covers the fallback branch itself."""
    worker = _fake_worker("m", None)
    env.manager.workers["m"] = worker
    conn = env.manager._send_tracked_task("m", ("TASK",))
    (worker_end, task), = worker.task_queue.items
    try:
        assert conn.poll(0.1) is False
        worker_end.send("token")
        assert conn.poll(1) is True and conn.recv() == "token"
        worker_end.close()
        assert conn.poll(1) is True
        with pytest.raises(EOFError):
            conn.recv()
    finally:
        worker_end.close()
        conn.close()


# --- load_and_begin_request on a model whose worker died --------------------------

def _registering_loader(env, calls):
    """A model loader like the server's load_model: registers a new live worker."""
    def loader(model_name, options):
        calls.append(model_name)
        env.manager.workers[model_name] = _live_fake_worker(model_name)
        return None
    return loader


def test_request_reloads_a_model_whose_worker_died(env):
    calls = []
    env.manager.set_model_loader(_registering_loader(env, calls))
    dead = _dead_fake_worker("m")
    env.manager.workers["m"] = dead

    assert env.manager.load_and_begin_request("m") is None

    assert calls == ["m"], f"the model with a dead worker was loaded {len(calls)} times"
    assert env.manager.workers["m"] is not dead, "the dead worker is still registered"
    assert env.manager._busy == {"m": 1}
    assert dead.process.joined and dead.task_queue.closed, "the dead worker was not cleaned up"


def test_request_on_a_live_worker_does_not_reload(env):
    """CONTROL: a live worker is used as it is."""
    calls = []
    env.manager.set_model_loader(_registering_loader(env, calls))
    live = _live_fake_worker("m")
    env.manager.workers["m"] = live

    assert env.manager.load_and_begin_request("m") is None

    assert calls == [], "a model with a live worker was loaded again"
    assert env.manager.workers["m"] is live
    assert env.manager._busy == {"m": 1}


@pytest.mark.parametrize("entry", ["request", "reaper"])
def test_gguf_worker_whose_load_started_no_process_is_dropped(env, entry):
    """A GGUF load whose every domain attempt returned -1 registers the worker
    with process == -1. It has no process to ask is_alive()/exitcode/join, so
    it counts as dead: a request loads it again once, the reaper removes it."""
    calls = []
    env.manager.set_model_loader(_registering_loader(env, calls))
    broken = _fake_worker("m", -1)
    env.manager.workers["m"] = broken

    if entry == "request":
        assert env.manager.load_and_begin_request("m") is None
        assert calls == ["m"], f"the model was loaded {len(calls)} times"
        assert env.manager.workers["m"] is not broken
        assert env.manager._busy == {"m": 1}
    else:
        env.manager.reap_dead_workers()
        assert "m" not in env.manager.workers, "the reaper kept a worker without a process"
        assert calls == []
    assert broken.task_queue.closed, "the worker without a process was not cleaned up"


# --- reaper -------------------------------------------------------------------------

def test_reaper_keeps_a_replacement_worker(env):
    """A load that replaced the dead worker between the reaper's check and its
    cleanup must keep its fresh worker (and its task queue)."""
    replacement = _live_fake_worker("m")

    def replace():
        env.manager.workers["m"] = replacement

    dead = _fake_worker("m", _FakeProcess(alive=False, exitcode=-signal.SIGKILL, on_is_alive=replace))
    env.manager.workers["m"] = dead

    env.manager.reap_dead_workers()

    assert env.manager.workers.get("m") is replacement, "the reaper removed the replacement worker"
    assert not replacement.task_queue.closed, "the reaper closed the replacement worker's queue"
    assert not replacement.process.joined
    assert dead.task_queue.closed, "the dead worker was not cleaned up"


def _held_by_another_thread(cond):
    """True if some thread other than a fresh probe thread holds cond's lock."""
    outcome = {}

    def probe():
        acquired = cond.acquire(blocking=False)
        if acquired:
            cond.release()
        outcome["held"] = not acquired

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "the lock probe never returned"
    return outcome["held"]


def test_reaper_drops_a_dead_worker_only_while_holding_the_workers_lock(env, monkeypatch):
    """Without the lock, a load could register a replacement between the
    reaper's check and its cleanup."""
    dead = _dead_fake_worker("m")
    env.manager.workers["m"] = dead
    real_drop = env.manager._drop_dead_worker_locked

    def drop_checking_the_lock(model_name, worker):
        assert _held_by_another_thread(env.manager._workers_cond), \
            "reap_dead_workers dropped a worker without holding the workers lock"
        real_drop(model_name, worker)

    monkeypatch.setattr(env.manager, "_drop_dead_worker_locked", drop_checking_the_lock)

    env.manager.reap_dead_workers()

    assert "m" not in env.manager.workers


def test_reaping_a_dead_worker_wakes_a_load_waiting_for_its_domain(env, monkeypatch):
    """Limit 1: a dead worker with a request still open holds the only NPU
    domain, so a load of another model waits. Reaping the dead worker frees the
    domain and must wake that load at once, not when its wait times out."""
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    monkeypatch.setattr(env.manager, "_domain_wait_timeout", lambda: 30)
    env.manager.workers["dead"] = _dead_fake_worker("dead")
    assert env.manager.begin_request("dead")

    cond = env.manager._workers_cond
    real_wait = cond.wait
    load_waits = threading.Event()

    def wait(timeout=None):
        load_waits.set()  # still holding the lock: the reaper gets it once real_wait releases it
        return real_wait(timeout)

    monkeypatch.setattr(cond, "wait", wait)
    outcome = {}
    load = threading.Thread(target=lambda: outcome.setdefault("result", env.manager.add_worker("b", "b.rkllm", "dir")),
                            daemon=True)
    load.start()
    assert load_waits.wait(5), "the load did not wait for the domain held by the dead worker"

    env.manager.reap_dead_workers()

    load.join(5)
    assert not load.is_alive(), "reaping the dead worker did not wake the waiting load"
    assert outcome == {"result": True}, outcome
    assert set(env.manager.workers) == {"b"}
    assert created[0].worker_model_info.base_domain_id == [1], "the load did not get the freed domain"


# --- abort and bookkeeping after the model was unloaded ----------------------------

@pytest.mark.parametrize("unload", ["removed", "replaced"])
def test_abort_after_unload_sets_the_original_workers_flag(env, unload):
    original = _live_fake_worker("m")
    env.manager.workers["m"] = original
    conn = env.manager._send_tracked_task("m", ("TASK",))
    replacement = _live_fake_worker("m")
    if unload == "removed":
        del env.manager.workers["m"]
    else:
        env.manager.workers["m"] = replacement

    try:
        conn.request_abort()
    finally:
        conn.close()

    assert original.abort_flag.value is True, "the abort did not reach the worker the request was sent to"
    assert replacement.abort_flag.value is False, "the abort hit another worker"


def test_update_expiration_of_an_unloaded_model_is_a_noop(env):
    env.manager.update_expiration_date_for_model("gone")
    env.manager.update_last_call_for_model("gone")
    assert env.manager.workers == {}
