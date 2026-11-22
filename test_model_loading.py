"""
Off-device regression tests for model loading in WorkerManager.

Load race: two simultaneous requests for a model that is not loaded yet (a
client such as Open WebUI sends the chat and its title/tags task together)
both loaded it. Both picked the same free NPU domain; either both failed
("failed to malloc npu memory") or both loaded and the second replaced the
first in WorkerManager.workers, leaving an untracked worker holding NPU
memory until the container restarted.

RAM check: the free-RAM test before a load added psutil's `free` to
`available`, which already includes it, so the least recently used model
was evicted too late or not at all.

NPU domains exhausted: with all max_number_models_loaded_in_memory domains
taken, the next load crashed with IndexError (base_domain_ids[0]). Now it
unloads the least recently used idle model, or waits while all are busy.
Running requests keep their model busy, so it is not unloaded mid-answer.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
import gc
import signal
import threading
import time
import types
from datetime import datetime, timedelta

import pytest

from test_worker_pipe_guard import _install_fakes, _load_worker_module

GB = 1_000_000_000


# --- helpers ------------------------------------------------------------------

def _fresh_worker_module():
    FakeRKLLM = _install_fakes()
    return FakeRKLLM, _load_worker_module()


@pytest.fixture
def env(monkeypatch):
    """Fresh fakes + worker module + a WorkerManager built by the REAL __init__
    (load lock, server SIGTERM handler), minus its host side effects: no orphan
    killing, no monitor thread. Restores signal handlers and empties the
    manager afterwards so its atexit stop_all() has nothing to stop."""
    FakeRKLLM, worker_mod = _fresh_worker_module()
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    monkeypatch.setattr(worker_mod, "_kill_orphaned_workers", lambda: None)
    monkeypatch.setattr(worker_mod.WorkerManager, "start_models_monitor", lambda self, interval=60: None)
    manager = worker_mod.WorkerManager()
    yield types.SimpleNamespace(FakeRKLLM=FakeRKLLM, worker_mod=worker_mod, manager=manager)
    manager.workers.clear()
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _use_fake_workers(env, monkeypatch, created, load_seconds=0.0, outcomes=None, size=0, ram_ok=True):
    """Replace Worker with a stand-in whose load only records itself, waits
    `load_seconds` and returns (or raises) the next scripted outcome."""
    outcomes = list(outcomes or [])

    class FakeWorker:
        def __init__(self, model_name, base_domain_id, loaded_by=None):
            self.worker_model_info = types.SimpleNamespace(
                model=model_name, size=size, base_domain_id=[base_domain_id], last_call=datetime.now())

        def create_worker_process(self, *args, **kwargs):
            created.append(self)
            time.sleep(load_seconds)
            outcome = outcomes.pop(0) if outcomes else True
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(env.worker_mod, "Worker", FakeWorker)
    monkeypatch.setattr(env.worker_mod, "is_rkllm_model", lambda name: True)
    monkeypatch.setattr(env.worker_mod, "is_gguf_model", lambda name: False)
    if ram_ok:
        monkeypatch.setattr(env.manager, "is_memory_available_for_model", lambda size: True)


def _run_concurrently(*calls, timeout=5):
    """Start all calls at the same instant (barrier) and return their results in call order."""
    start = threading.Barrier(len(calls))
    results = [None] * len(calls)

    def runner(i, call):
        start.wait()
        results[i] = call()

    # daemon: a call stuck on a leaked lock fails the test instead of hanging the test run
    threads = [threading.Thread(target=runner, args=(i, c), daemon=True) for i, c in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a concurrent call never returned (deadlock?)"
    return results


# --- load lock ------------------------------------------------------------------

def test_concurrent_loads_create_one_worker(env, monkeypatch):
    created = []
    _use_fake_workers(env, monkeypatch, created, load_seconds=0.3)  # both requests overlap the load

    results = _run_concurrently(
        lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"),
        lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"),
    )

    assert len(created) == 1, f"model loaded {len(created)} times for 2 simultaneous requests"
    assert results == [True, True], results
    assert env.manager.workers["qwen"] is created[0]


def test_concurrent_loads_of_different_models_both_load_on_distinct_domains(env, monkeypatch):
    """The lock serializes loads, it must not refuse them; and the second load
    must see the first one registered so it gets a different NPU domain."""
    created = []
    _use_fake_workers(env, monkeypatch, created, load_seconds=0.2)

    results = _run_concurrently(
        lambda: env.manager.add_worker("qwen-2b", "a.rkllm", "dir-a"),
        lambda: env.manager.add_worker("qwen-4b", "b.rkllm", "dir-b"),
    )

    assert results == [True, True], results
    assert set(env.manager.workers) == {"qwen-2b", "qwen-4b"}
    domains = [w.worker_model_info.base_domain_id[0] for w in env.manager.workers.values()]
    assert len(set(domains)) == 2, f"both models were given NPU domain {domains[0]}"


def test_failed_load_registers_nothing_and_next_load_proceeds(env, monkeypatch):
    created = []
    _use_fake_workers(env, monkeypatch, created,
                      outcomes=[False, RuntimeError("load crashed"), True])

    assert env.manager.add_worker("qwen", "model.rkllm", "dir") is False
    assert "qwen" not in env.manager.workers, "a failed load was registered"

    with pytest.raises(RuntimeError, match="load crashed"):
        env.manager.add_worker("qwen", "model.rkllm", "dir")
    assert "qwen" not in env.manager.workers, "a crashed load was registered"

    # Runs in a thread: if the exception above left the lock held, this hangs
    (result,) = _run_concurrently(lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"))
    assert result is True
    assert env.manager.workers["qwen"] is created[2]


# --- RAM check and least-recently-used eviction -----------------------------------

def test_memory_check_does_not_count_free_ram_twice(monkeypatch):
    """psutil's `available` (MemAvailable) already includes `free` (MemFree).
    Adding them made a 5.4 GB load look like it fits into 4.0 GB available,
    so no model was evicted and the load could run the box out of RAM."""
    _, worker_mod = _fresh_worker_module()
    fake_vm = types.SimpleNamespace(available=4 * GB, free=int(2.5 * GB))
    monkeypatch.setattr(worker_mod.psutil, "virtual_memory", lambda: fake_vm)
    manager = worker_mod.WorkerManager.__new__(worker_mod.WorkerManager)  # the check uses no instance state

    model_size = int(4.5 * GB)  # needs 4.5 * 1.2 = 5.4 GB
    assert not manager.is_memory_available_for_model(model_size), "5.4 GB needed, only 4.0 GB available"
    assert manager.is_memory_available_for_model(3 * GB), "3.6 GB needed, 4.0 GB available"


def test_add_worker_evicts_least_recently_used_model_until_ram_fits(env, monkeypatch):
    # 3 GB really available (2 GB of it free); the new model needs 3.5 * 1.2 = 4.2 GB.
    # Evicting the oldest model (2 GB) is enough; the double-counting formula
    # (3 + 2 = 5 GB) would have evicted nothing.
    fake_vm = types.SimpleNamespace(available=3 * GB, free=2 * GB)
    monkeypatch.setattr(env.worker_mod.psutil, "virtual_memory", lambda: fake_vm)
    monkeypatch.setattr(env.worker_mod.time, "sleep", lambda seconds: None)  # eviction waits 1 s per model

    now = datetime.now()
    for name, minutes_ago, size, domain in [("newest", 1, 1 * GB, 7), ("oldest", 30, 2 * GB, 9), ("middle", 10, 1 * GB, 8)]:
        env.manager.workers[name] = types.SimpleNamespace(worker_model_info=types.SimpleNamespace(
            model=name, size=size, base_domain_id=[domain], last_call=now - timedelta(minutes=minutes_ago)))

    evicted = []

    def fake_stop_worker(model_name, timeout=30):
        evicted.append(model_name)
        fake_vm.available += env.manager.workers.pop(model_name).worker_model_info.size

    monkeypatch.setattr(env.manager, "stop_worker", fake_stop_worker)
    created = []
    _use_fake_workers(env, monkeypatch, created, size=int(3.5 * GB), ram_ok=False)

    assert env.manager.add_worker("qwen-4b", "4b.rkllm", "dir") is True
    assert evicted == ["oldest"], f"evicted {evicted}, expected only the least recently used model"
    assert set(env.manager.workers) == {"newest", "middle", "qwen-4b"}


# --- NPU domains (max_number_models_loaded_in_memory) ------------------------------

def _limit_models(env, monkeypatch, n):
    """Set max_number_models_loaded_in_memory (= number of NPU domains for LLMs) to n."""
    real_get = env.worker_mod.rkllama.config.get

    def get(section, key, *args, **kwargs):
        if (section, key) == ("model", "max_number_models_loaded_in_memory"):
            return str(n)
        return real_get(section, key, *args, **kwargs)

    monkeypatch.setattr(env.worker_mod.rkllama.config, "get", get)


class _ListQueue:
    """Stands in for a worker task queue: records what the manager sends."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        pass


def _loaded(env, name, domain, minutes_ago):
    """Register an already loaded worker that the real stop_worker can unload."""
    env.manager.workers[name] = types.SimpleNamespace(
        process=None, task_queue=_ListQueue(), abort_flag=types.SimpleNamespace(value=False),
        worker_model_info=types.SimpleNamespace(
            model=name, size=0, base_domain_id=[domain],
            last_call=datetime.now() - timedelta(minutes=minutes_ago),
            expires_at=datetime.now() + timedelta(minutes=30)))


def test_domain_allocator_offers_every_domain_up_to_the_limit(env, monkeypatch):
    _limit_models(env, monkeypatch, 3)
    assert env.manager.free_base_domain_ids(highest_first=True) == [3, 2, 1]
    assert env.manager.free_base_domain_ids(highest_first=False) == [1, 2, 3]
    _loaded(env, "a", 2, 1)
    assert env.manager.free_base_domain_ids(highest_first=False) == [1, 3]


# --- full NPU domains: unload the least recently used idle model, wait while all are busy

def _start_load(env, name):
    """add_worker in a daemon thread; returns (thread, outcome dict)."""
    outcome = {}

    def load():
        try:
            outcome["result"] = env.manager.add_worker(name, f"{name}.rkllm", "dir")
        except Exception as e:  # noqa: BLE001 - the test inspects it
            outcome["error"] = e

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    return thread, outcome


def test_full_domains_unload_least_recently_used_idle_llm(env, monkeypatch):
    _limit_models(env, monkeypatch, 2)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "tts", 0, 120)       # RKNN model on domain 0: oldest, but holds no LLM domain
    _loaded(env, "old", 2, 30)
    _loaded(env, "new", 1, 1)

    assert env.manager.add_worker("third", "third.rkllm", "dir") is True

    assert set(env.manager.workers) == {"tts", "new", "third"}, set(env.manager.workers)
    assert created[0].worker_model_info.base_domain_id == [2], "the new model did not get the freed domain"


def test_limit_one_swaps_models(env, monkeypatch):
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "qwen-2b", 1, 5)

    assert env.manager.add_worker("qwen-4b", "4b.rkllm", "dir") is True

    assert set(env.manager.workers) == {"qwen-4b"}
    assert created[0].worker_model_info.base_domain_id == [1]


def test_busy_model_is_not_unloaded_even_if_least_recently_used(env, monkeypatch):
    _limit_models(env, monkeypatch, 2)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "old-busy", 2, 30)
    _loaded(env, "new-idle", 1, 1)
    assert env.manager.begin_request("old-busy")

    assert env.manager.add_worker("third", "third.rkllm", "dir") is True

    assert set(env.manager.workers) == {"old-busy", "third"}, "a model with a running request was unloaded"


def test_load_waits_while_all_models_are_busy_then_proceeds(env, monkeypatch):
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "a", 1, 5)
    assert env.manager.begin_request("a")

    thread, outcome = _start_load(env, "b")
    thread.join(0.3)
    assert thread.is_alive(), f"load did not wait for the busy model: {outcome}"
    assert set(env.manager.workers) == {"a"}

    env.manager.end_request("a")  # the running answer finishes
    thread.join(5)
    assert not thread.is_alive(), "load did not wake up when the busy model became idle"
    assert outcome == {"result": True}, outcome
    assert set(env.manager.workers) == {"b"}


def test_unloading_a_model_wakes_a_waiting_load(env, monkeypatch):
    """A domain freed by the monitor (expired model) or the unload API must wake a waiting load."""
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "a", 1, 5)
    assert env.manager.begin_request("a")

    thread, outcome = _start_load(env, "b")
    thread.join(0.3)
    assert thread.is_alive(), f"load did not wait for the busy model: {outcome}"

    env.manager.stop_worker("a")
    thread.join(5)
    assert not thread.is_alive(), "load did not wake up when a model was unloaded"
    assert outcome == {"result": True}, outcome


def test_load_gives_up_with_clear_error_when_all_models_stay_busy(env, monkeypatch):
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    monkeypatch.setattr(env.manager, "_domain_wait_timeout", lambda: 0.2)
    _loaded(env, "a", 1, 5)
    assert env.manager.begin_request("a")

    with pytest.raises(RuntimeError, match="model slots"):
        env.manager.add_worker("b", "b.rkllm", "dir")

    assert set(env.manager.workers) == {"a"}, "the busy model was unloaded"
    assert created == [], "the new model was loaded anyway"


def test_two_waiting_loads_of_one_model_load_it_once(env, monkeypatch):
    """Open WebUI sends the chat and its title task together. With all slots
    busy both loads wait; when a slot frees up, the first loads the model and
    the second must use it, not unload it to load it again (log 2026-11-13)."""
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    _loaded(env, "busy", 1, 5)
    assert env.manager.begin_request("busy")

    first, first_outcome = _start_load(env, "qwen-0.8b")
    second, second_outcome = _start_load(env, "qwen-0.8b")
    first.join(0.3)
    assert first.is_alive() and second.is_alive(), "the loads did not wait for the busy model"

    env.manager.end_request("busy")
    first.join(5)
    second.join(5)

    assert first_outcome == second_outcome == {"result": True}, (first_outcome, second_outcome)
    assert len(created) == 1, f"the model was loaded {len(created)} times"
    assert set(env.manager.workers) == {"qwen-0.8b"}


# --- running requests keep their model busy -----------------------------------------

def test_inference_pipe_keeps_model_busy_until_closed(env, monkeypatch):
    _loaded(env, "m", 1, 5)

    conn = env.manager.inference("m", [1, 2, 3], None, {})
    assert env.manager.is_busy("m"), "a running inference does not mark the model busy"
    assert env.manager.workers["m"].task_queue.items, "the task never reached the worker queue"

    conn.close()
    assert not env.manager.is_busy("m")


def test_closing_a_pipe_twice_does_not_release_another_running_request(env, monkeypatch):
    """Readers may close twice (error path, then timeout path). A second release
    would mark the model idle while another request on it still runs, so a swap
    could unload it mid-answer."""
    _loaded(env, "m", 1, 5)
    first = env.manager.inference("m", [1], None, {})
    second = env.manager.inference("m", [2], None, {})

    first.close()
    first.close()
    assert env.manager.is_busy("m"), "double close released the other running request"

    second.close()
    assert not env.manager.is_busy("m")


def test_abandoned_pipe_releases_model_when_garbage_collected(env, monkeypatch):
    """A reader that never closes (client gone mid-stream, exception) must not
    pin the model as busy forever: it would block every later swap."""
    _loaded(env, "m", 1, 5)

    conn = env.manager.inference("m", [1, 2, 3], None, {})
    assert env.manager.is_busy("m")
    del conn
    gc.collect()
    assert not env.manager.is_busy("m"), "an abandoned pipe kept the model busy"


def test_request_on_unloaded_model_fails_cleanly(env, monkeypatch):
    with pytest.raises(RuntimeError, match="not loaded"):
        env.manager.inference("gone", [1], None, {})
    assert env.manager._busy == {}
    assert env.manager.begin_request("gone") is False


def test_failure_before_send_releases_model(env, monkeypatch):
    """E.g. the vision encoder fails before a multimodal task is sent."""
    _loaded(env, "m", 1, 5)

    def encoder_fails():
        raise RuntimeError("Unexpected error encoding image")

    with pytest.raises(RuntimeError, match="encoding image"):
        env.manager._send_tracked_task("m", lambda prepared: ("TASK", prepared), before_send=encoder_fails)
    assert not env.manager.is_busy("m")
    assert env.manager.workers["m"].task_queue.items == [], "a task was sent although its preparation failed"


# --- load_and_begin_request: load if needed and mark busy in one step ------------------------------------

def _recording_loader(env, calls, error=None):
    """A model loader like the server's load_model: loads through add_worker."""
    def loader(model_name, options):
        calls.append((model_name, options))
        if error:
            return error
        env.manager.add_worker(model_name, f"{model_name}.rkllm", "dir", options=options)
        return None
    return loader


def test_acquire_marks_a_loaded_model_busy_without_loading_it(env, monkeypatch):
    calls = []
    env.manager.set_model_loader(_recording_loader(env, calls))
    _loaded(env, "m", 1, 5)

    assert env.manager.load_and_begin_request("m") is None
    assert env.manager.is_busy("m")
    assert calls == [], "an already loaded model was loaded again"


def test_acquire_loads_a_model_that_is_not_loaded_and_marks_it_busy(env, monkeypatch):
    """E.g. the model was unloaded for another one after the route loaded it:
    the request loads it again instead of failing with 503."""
    created = []
    _use_fake_workers(env, monkeypatch, created)
    calls = []
    env.manager.set_model_loader(_recording_loader(env, calls))

    assert env.manager.load_and_begin_request("m", {"num_ctx": "4096"}) is None
    assert calls == [("m", {"num_ctx": "4096"})], "the request options did not reach the loader"
    assert "m" in env.manager.workers and env.manager.is_busy("m")


def test_acquire_reports_a_failed_load_and_leaves_nothing_busy(env, monkeypatch):
    env.manager.set_model_loader(_recording_loader(env, [], error="Modelfile not found"))

    assert env.manager.load_and_begin_request("m") == "Modelfile not found"
    assert env.manager._busy == {}


def test_acquired_model_cannot_be_unloaded_before_its_request_starts(env, monkeypatch):
    """Load and busy mark happen in one hold of the lock: another model's load
    that comes in right after the load (limit 1) must not unload the fresh model
    before the request marks it busy (log 2026-11-13: HTTP 503)."""
    _limit_models(env, monkeypatch, 1)
    created = []
    _use_fake_workers(env, monkeypatch, created)
    monkeypatch.setattr(env.manager, "_domain_wait_timeout", lambda: 0.5)
    loaded = threading.Event()

    def loader(model_name, options):
        env.manager.add_worker(model_name, f"{model_name}.rkllm", "dir")
        loaded.set()
        time.sleep(0.2)  # the window in which a competing load would run
        return None

    env.manager.set_model_loader(loader)
    outcome = {}
    acquirer = threading.Thread(target=lambda: outcome.setdefault("a", env.manager.load_and_begin_request("a")), daemon=True)
    acquirer.start()
    assert loaded.wait(5)
    competitor, competitor_outcome = _start_load(env, "b")

    acquirer.join(5)
    competitor.join(5)
    assert outcome == {"a": None}, outcome
    assert env.manager.is_busy("a"), "the acquired model is not busy"
    assert set(env.manager.workers) == {"a"}, "the fresh model was unloaded before its request started"
    assert isinstance(competitor_outcome.get("error"), env.worker_mod.ModelSlotsBusy), competitor_outcome
