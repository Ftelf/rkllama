"""
Off-device tests for issue #29, server side: the start and end line of a request.

One request to an rkllm model (chat, generate, embed) logs one start line and one
end line with the same id (logger ``rkllama.request``), whatever its outcome.
Log lines only: no test here may need a behaviour change. The worker and manager
lines are in test_request_log_worker.py.

The REAL server_utils.py runs with a scripted fake worker pipe (test_tool_calls.py), so
every outcome is exercised in each of the four pipe-reading loops and the embed loop.
"""
import logging
import re
import sys

import pytest

from test_tool_calls import _server_utils

FINISHED = "<RKLLM_TASK_FINISHED>"
ERROR_SENTINEL = "<RKLLM_TASK_ERROR>"

START = re.compile(r"request (?P<id>[0-9a-f]{8}) start endpoint=(?P<endpoint>\w+) api=(?P<api>\w+) model=(?P<model>.*)")
END = re.compile(r"request (?P<id>[0-9a-f]{8}) end outcome=(?P<outcome>[a-z ]+) duration_ms=(?P<ms>\d+) tokens=(?P<tokens>\S+)")


# --- the server side: start and end lines -----------------------------------------------

class ScriptedPipe:
    """A worker pipe that replays ``tokens`` and then ``ending``; ``ending`` is one of
    finished / aborted / error / timeout / dead, or a ready-made final message."""

    def __init__(self, tokens, ending, final=(FINISHED, 7, 5, 10, 20)):
        self.messages = list(tokens)
        self.ending = ending
        self.final = final
        self.worker_gone = False
        self.aborts = 0
        self.closed = False

    def poll(self, timeout=None):
        if self.messages:
            return True
        return self.ending != "timeout"

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        if self.ending == "dead":
            self.worker_gone = True
            raise EOFError("the worker exited")
        if self.ending == "error":
            return (ERROR_SENTINEL, "the model failed")
        if self.ending == "aborted":
            return (FINISHED, 0, 0, 0, 0)
        return self.final

    def request_abort(self):
        self.aborts += 1

    def close(self):
        self.closed = True


class ScriptedManager:
    def __init__(self, tokens, ending, final=(FINISHED, 7, 5, 10, 20), load_error=None, context_length=None):
        self.tokens = tokens
        self.ending = ending
        self.final = final
        self.load_error = load_error
        self.context_length = context_length
        self.request_ids = []
        self.loads = []
        self.ended = 0

    def load_and_begin_request(self, model_name, options=None, request_id=None):
        self.loads.append(request_id)
        if isinstance(self.load_error, BaseException):
            raise self.load_error
        return self.load_error

    def end_request(self, model_name):
        self.ended += 1

    def loaded_context_length(self, model_name):
        return self.context_length

    def update_expiration_date_for_model(self, model_name):
        pass

    def inference(self, model_name, prompt, prompt_cache_file, options, request_id=None):
        self.request_ids.append(request_id)
        return ScriptedPipe(self.tokens, self.ending, self.final)

    def embedding(self, model_name, text, prompt_cache_file, options, request_id=None):
        self.request_ids.append(request_id)
        return ScriptedPipe([], self.ending, self.final)


class FakeArray:
    def tolist(self):
        return [0.1, 0.2]


class EmbeddingPipe(ScriptedPipe):
    def recv(self):
        if self.ending == "dead":
            self.worker_gone = True
            raise EOFError("the worker exited")
        return {"embedding": FakeArray(), "num_tokens": 3}


class EmbeddingManager(ScriptedManager):
    def embedding(self, model_name, text, prompt_cache_file, options, request_id=None):
        self.request_ids.append(request_id)
        return EmbeddingPipe([], self.ending)


def request_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "rkllama.request"]


def one_start_and_one_end(caplog):
    """(start, end) of the one request that ran: exactly one of each, the same id."""
    lines = request_lines(caplog)
    starts = [START.fullmatch(line) for line in lines if " start " in line]
    ends = [END.fullmatch(line) for line in lines if " end " in line]
    assert len(starts) == 1 and all(starts), lines
    assert len(ends) == 1 and all(ends), lines
    assert len(lines) == 2, lines
    assert starts[0]["id"] == ends[0]["id"], lines
    assert lines[0] == starts[0].string, "the start line must come first"
    return starts[0], ends[0]


def chat(sm, manager, stream, tokens=("a", "b"), options=None, **kwargs):
    sm.variables.worker_manager_rkllm = manager
    return sm.ChatEndpointHandler.handle_request(
        "qwen-test", [{"role": "user", "content": "q"}], stream=stream, options=options, **kwargs)


def generate(sm, manager, stream, options=None, **kwargs):
    sm.variables.worker_manager_rkllm = manager
    return sm.GenerateEndpointHandler.handle_request("qwen-test", "q", stream=stream, options=options, **kwargs)


def drain(result, stream):
    """Read a response to its end (a stream is a generator that only runs when read)."""
    if stream:
        return list(result.get_data())
    return result


LOOPS = [("chat", False), ("chat", True), ("generate", False), ("generate", True)]
LOOP_IDS = [f"{name}-{'stream' if stream else 'complete'}" for name, stream in LOOPS]


@pytest.fixture
def sm(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    module = _server_utils(monkeypatch, [])
    module.variables = sys.modules["rkllama.api.variables"]
    return module


def run_loop(sm, name, stream, manager, options=None):
    handler = chat if name == "chat" else generate
    result = handler(sm, manager, stream, options=options)
    return drain(result, stream)


@pytest.mark.parametrize("name, stream", LOOPS, ids=LOOP_IDS)
@pytest.mark.parametrize("case, ending, final, options, outcome, tokens", [
    ("finished", "finished", (FINISHED, 7, 5, 10, 20), None, "finished", "5"),
    ("token limit", "finished", (FINISHED, 7, 5, 10, 20), {"max_new_tokens": "5"}, "token limit", "5"),
    ("below the token limit", "finished", (FINISHED, 7, 4, 10, 20), {"max_new_tokens": "5"}, "finished", "4"),
    ("limit 0 means the runtime default: no claim", "finished", (FINISHED, 7, 5, 10, 20), {"max_new_tokens": "0"}, "finished", "5"),
    ("aborted", "aborted", None, None, "aborted", "0"),
    ("worker error", "error", None, None, "error", "-"),
    ("worker dead", "dead", None, None, "worker dead", "-"),
])
def test_one_start_and_one_end_line_per_outcome(sm, caplog, name, stream, case, ending, final, options, outcome, tokens):
    manager = ScriptedManager([], ending, final or (FINISHED, 0, 0, 0, 0))
    # the answer tokens: count them so "tokens" can be checked where the worker did not say
    manager.tokens = ["x"] * int(tokens) if tokens.isdigit() and ending == "finished" else []
    run_loop(sm, name, stream, manager, options)

    start, end = one_start_and_one_end(caplog)
    assert end["outcome"] == outcome, request_lines(caplog)
    assert end["tokens"] == tokens, request_lines(caplog)
    assert start["endpoint"] == name and start["api"] == "ollama" and start["model"] == "qwen-test"
    assert manager.request_ids == [start["id"]], "the worker task did not get the request id"
    assert manager.ended == 1, "the model was not released"


@pytest.mark.parametrize("name, stream", LOOPS, ids=LOOP_IDS)
def test_timeout_is_an_outcome_and_aborts_the_run(sm, caplog, name, stream):
    manager = ScriptedManager(["a", "b"], "timeout")
    run_loop(sm, name, stream, manager)

    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "timed out", request_lines(caplog)


@pytest.mark.parametrize("name", ["chat", "generate"])
def test_client_that_leaves_a_stream_is_logged_once_with_the_tokens_seen(sm, caplog, name):
    manager = ScriptedManager(["a"] * 50, "finished")
    handler = chat if name == "chat" else generate
    response = handler(sm, manager, True)
    stream = response.get_data()
    next(stream)
    next(stream)
    stream.close()          # what the server does when the client closes its connection

    start, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "client gone", request_lines(caplog)
    assert end["tokens"].isdigit() and 0 < int(end["tokens"]) < 50, request_lines(caplog)
    # a second close() or a late garbage collection must not add a line
    stream.close()
    assert len(request_lines(caplog)) == 2


def test_a_stream_that_was_read_to_its_end_is_not_a_client_gone(sm, caplog):
    manager = ScriptedManager(["a", "b"], "finished")
    response = chat(sm, manager, True)
    stream = response.get_data()
    list(stream)
    stream.close()

    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "finished", request_lines(caplog)


def test_end_line_of_a_stream_comes_when_it_is_over_not_when_the_handler_returns(sm, caplog):
    manager = ScriptedManager(["a", "b"], "finished")
    response = chat(sm, manager, True)
    assert [line for line in request_lines(caplog) if " end " in line] == []
    list(response.get_data())
    assert len([line for line in request_lines(caplog) if " end " in line]) == 1


def test_openai_requests_are_marked(sm, caplog):
    chat(sm, ScriptedManager(["a"], "finished"), False, is_openai_request=True)
    start, _ = one_start_and_one_end(caplog)
    assert start["api"] == "openai"


@pytest.mark.parametrize("ending, outcome", [("finished", "finished"), ("timeout", "timed out")])
def test_embed_logs_one_start_and_one_end(sm, caplog, ending, outcome):
    manager = EmbeddingManager([], ending)
    sm.variables.worker_manager_rkllm = manager
    if ending == "timeout":
        # a timed out embedding has no tensor: the handler fails today (not a #29 change); the request still ends once
        with pytest.raises(AttributeError):
            sm.EmbedEndpointHandler.handle_request("qwen-test", "text")
    else:
        sm.EmbedEndpointHandler.handle_request("qwen-test", "text")

    start, end = one_start_and_one_end(caplog)
    assert end["outcome"] == outcome, request_lines(caplog)
    assert start["endpoint"] == "embed"
    assert manager.request_ids == [start["id"]]


def test_embed_with_a_dead_worker_ends_as_worker_dead(sm, caplog):
    manager = EmbeddingManager([], "dead")
    sm.variables.worker_manager_rkllm = manager
    with pytest.raises(EOFError):
        sm.EmbedEndpointHandler.handle_request("qwen-test", "text")
    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "worker dead", request_lines(caplog)
    assert manager.ended == 1


@pytest.mark.parametrize("load_error, status", [("no memory", 500), ("busy", 503)])
def test_a_request_that_cannot_load_its_model_still_gets_both_lines(sm, caplog, load_error, status):
    if status == 503:
        load_error = sm.ModelSlotsBusy("all model slots are taken")
    manager = ScriptedManager([], "finished", load_error=load_error)
    response, code = chat(sm, manager, False)

    assert code == status
    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "error"
    assert manager.request_ids == [], "no task may be sent for a model that did not load"
    assert manager.ended == 0, "the model was never marked busy"


def test_a_prompt_rejected_for_its_length_gets_both_lines(sm, caplog):
    manager = ScriptedManager([], "finished", context_length=2)   # the stubbed prompt has 3 tokens
    response, code = chat(sm, manager, False)

    assert code == 400
    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "error"
    assert manager.request_ids == []
    assert manager.ended == 1


def test_an_exception_in_the_handler_ends_the_request_as_error(sm, caplog, monkeypatch):
    manager = ScriptedManager([], "finished")
    monkeypatch.setattr(sm.EndpointHandler, "prepare_prompt",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tokenizer failed"))))
    with pytest.raises(RuntimeError):
        chat(sm, manager, False)
    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "error"
    assert manager.ended == 1


def test_each_request_gets_its_own_id(sm, caplog):
    for _ in range(20):
        chat(sm, ScriptedManager(["a"], "finished"), False)
    ids = [START.fullmatch(line)["id"] for line in request_lines(caplog) if " start " in line]
    assert len(ids) == 20 and len(set(ids)) == 20


def test_a_model_name_cannot_forge_a_log_line(sm, caplog):
    manager = ScriptedManager(["a"], "finished")
    sm.variables.worker_manager_rkllm = manager
    sm.ChatEndpointHandler.handle_request("evil\nrequest deadbeef end outcome=finished duration_ms=1 tokens=1",
                                          [{"role": "user", "content": "q"}], stream=False)
    lines = request_lines(caplog)
    assert len(lines) == 2 and all("\n" not in line for line in lines), lines


def test_trace_end_writes_once_even_from_two_threads(caplog):
    import threading
    from rkllama.api.request_log import RequestTrace
    caplog.set_level(logging.INFO)
    trace = RequestTrace("chat", "m")
    trace.start()
    barrier = threading.Barrier(8)

    def end():
        barrier.wait()
        trace.end("finished")

    threads = [threading.Thread(target=end) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    one_start_and_one_end(caplog)


def test_trace_without_an_outcome_ends_as_error(caplog):
    from rkllama.api.request_log import RequestTrace
    caplog.set_level(logging.INFO)
    trace = RequestTrace("chat", "m")
    trace.start()
    trace.end()
    _, end = one_start_and_one_end(caplog)
    assert end["outcome"] == "error" and end["tokens"] == "-"


# --- no user content in the new lines ---------------------------------------------------

MARKER = "SECRET-MARKER-4711"


@pytest.mark.parametrize("name, stream", LOOPS, ids=LOOP_IDS)
@pytest.mark.parametrize("ending", ["finished", "aborted", "error", "timeout", "dead"])
def test_no_prompt_or_answer_text_reaches_the_log(sm, caplog, monkeypatch, name, stream, ending):
    monkeypatch.setattr(sm.EndpointHandler, "prepare_prompt",
                        staticmethod(lambda *a, **k: ([1, 2, 3], f"cache-{MARKER}", f"prompt {MARKER}")))
    manager = ScriptedManager([f"answer {MARKER}"], ending)
    sm.variables.worker_manager_rkllm = manager
    if name == "chat":
        result = sm.ChatEndpointHandler.handle_request(
            "qwen-test", [{"role": "user", "content": f"q {MARKER}"}], system=f"system {MARKER}", stream=stream,
            options={"stop": [MARKER]})
    else:
        result = sm.GenerateEndpointHandler.handle_request(
            "qwen-test", f"q {MARKER}", system=f"system {MARKER}", stream=stream, options={"stop": [MARKER]})
    body = drain(result, stream)

    leaked = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO and MARKER in r.getMessage()]
    assert leaked == [], leaked
    if ending in ("finished", "aborted", "timeout"):
        # Positive control: the marker went through the handler, the answer to the client carries it
        sent = "".join(body) if stream else body[0].get_data().decode()
        assert MARKER in sent, "the marker never reached the response, so the check above proves nothing"
