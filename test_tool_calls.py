"""
Off-device tests for tool-call parsing (issue #28).

The parser (format_utils.get_tool_calls) is loaded by file path under a private
module name, with its heavy imports (flask, cv2, numpy, PIL) stubbed; nothing in
it needs them for tool calls. The chat handler tests load the REAL server_utils.py
with the REAL format_utils.py behind it, so the whole non-stream / stream / OpenAI
path runs as on the board; only the model worker is a fake pipe that replays a
fixed answer token by token (fake-module harness of test_worker_pipe_guard.py).

The first part pins down how today's JSON tool calls are parsed and answered
(characterisation, green before the XML parser was added): the XML change must
not move any of it.
"""
import importlib.util
import json
import sys
import types

import pytest

from test_worker_pipe_guard import SRC, _install_fakes, _load_worker_module

FORMAT_UTILS = SRC / "rkllama" / "api" / "format_utils.py"
SERVER_UTILS = SRC / "rkllama" / "api" / "server_utils.py"


# --- harness ---------------------------------------------------------------------------

class FakeResponse:
    """Stands in for flask.Response: keeps the body (bytes, or a generator for streams)."""
    def __init__(self, response=None, status=None, mimetype=None, content_type=None, **kwargs):
        self.response = response

    def get_data(self):
        return self.response


def _jsonify(obj):
    return FakeResponse(json.dumps(obj).encode("utf-8"))


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load_format_utils(monkeypatch, module_name="_test_tool_calls_format_utils"):
    """Load the REAL format_utils.py. Under a private name by default, so the
    `rkllama.api.format_utils` stub of other test files is never replaced."""
    _stub_module(monkeypatch, "flask", jsonify=_jsonify, Response=FakeResponse, stream_with_context=lambda gen: gen)
    _stub_module(monkeypatch, "cv2")
    _stub_module(monkeypatch, "numpy")
    pil_image = _stub_module(monkeypatch, "PIL.Image", Image=type("Image", (), {}))  # annotations use Image.Image
    _stub_module(monkeypatch, "PIL", Image=pil_image)
    spec = importlib.util.spec_from_file_location(module_name, str(FORMAT_UTILS))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fu(monkeypatch):
    return _load_format_utils(monkeypatch)


class FakePipe:
    """A worker pipe that returns the answer tokens, then the finished tuple."""
    def __init__(self, tokens):
        self.messages = list(tokens) + [("<RKLLM_TASK_FINISHED>", 7, len(tokens), 10, 20)]

    def poll(self, timeout=None):
        return True

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        pass

    def request_abort(self):
        pass


class FakeManager:
    def __init__(self, tokens):
        self.tokens = tokens

    def load_and_begin_request(self, model_name, options=None):
        return None

    def end_request(self, model_name):
        pass

    def loaded_context_length(self, model_name):
        return None

    def update_expiration_date_for_model(self, model_name):
        pass

    def inference(self, model_name, prompt, prompt_cache_file, options):
        return FakePipe(self.tokens)


def _server_utils(monkeypatch, tokens):
    """The REAL server_utils.py on top of the REAL format_utils.py; the model answers `tokens`."""
    _install_fakes()
    _load_format_utils(monkeypatch, "rkllama.api.format_utils")   # restored after the test
    _stub_module(monkeypatch, "transformers", AutoTokenizer=object)
    _stub_module(monkeypatch, "rkllama.api.variables", system="", global_status=-1,
                 worker_manager_rkllm=FakeManager(tokens))
    _load_worker_module()
    spec = importlib.util.spec_from_file_location("rkllama.api.server_utils", str(SERVER_UTILS))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)   # the stream replays held tokens with 0.1 s gaps
    monkeypatch.setattr(mod.EndpointHandler, "prepare_prompt",
                        staticmethod(lambda *args, **kwargs: ([1, 2, 3], "prompt-cache-file", "prompt")))
    return mod


def _chat(monkeypatch, tokens, tools, stream=False, openai=False):
    """Run one /api/chat (or /v1/chat/completions) request through ChatEndpointHandler.
    Non-stream: the response body as a dict. Stream: the list of chunks (dicts for
    Ollama NDJSON lines, dicts of the `data:` payloads for OpenAI SSE)."""
    server_utils = _server_utils(monkeypatch, tokens)
    result = server_utils.ChatEndpointHandler.handle_request(
        "qwen-test", [{"role": "user", "content": "q"}], stream=stream, tools=tools, is_openai_request=openai)
    if not stream:
        response, code = result
        assert code == 200
        return json.loads(response.get_data())
    lines = [line for line in result.response]
    if openai:
        payloads = [line[len("data: "):].strip() for line in lines]
        return [json.loads(p) for p in payloads if p != "[DONE]"]
    return [json.loads(line) for line in lines]


WEATHER_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

JSON_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
PARIS_CALL = [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]


# --- characterisation: today's JSON tool-call parsing -----------------------------------

@pytest.mark.parametrize("answer, expected", [
    (JSON_CALL, PARIS_CALL),
    # "parameters" (Llama chat template) is renamed to "arguments"
    ('<tool_call>\n{"name": "get_weather", "parameters": {"city": "Paris"}}\n</tool_call>', PARIS_CALL),
    # words around the JSON inside the tags
    ('<tool_call> Output : {"name": "get_weather", "arguments": {"city": "Paris"}} </tool_call>', PARIS_CALL),
    # no tags at all (Llama 3.2): the whole answer is searched
    ('Sure. {"name": "get_weather", "arguments": {"city": "Paris"}}', PARIS_CALL),
    # two blocks, in text order; other keys of the JSON object are kept
    ('<tool_call>{"name": "a", "arguments": {}}</tool_call>\n<tool_call>{"name": "b", "arguments": {"x": 1}, "id": 3}</tool_call>',
     [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {"x": 1}, "id": 3}}]),
    # two JSON objects in one block
    ('<tool_call>{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {}}</tool_call>',
     [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]),
    # a block without a call does not stop the whole-text search, which finds nothing here
    ('<tool_call>not json</tool_call>', []),
    # JSON objects without name + arguments/parameters are not calls
    ('{"name": "get_weather"} {"arguments": {}}', []),
    ('The weather in Paris is sunny.', []),
    ('', []),
])
def test_json_tool_call_forms(fu, answer, expected):
    assert fu.get_tool_calls(answer) == expected


def test_json_chat_complete_has_empty_content_and_tool_calls(monkeypatch):
    body = _chat(monkeypatch, [JSON_CALL[:11], JSON_CALL[11:]], WEATHER_TOOLS)

    assert body["message"]["tool_calls"] == PARIS_CALL
    assert body["message"]["content"] == ""
    assert body["done_reason"] == "tool_calls"


def test_json_chat_complete_keeps_no_text_around_the_call(monkeypatch):
    body = _chat(monkeypatch, ["Let me check.\n", JSON_CALL, "\nDone."], WEATHER_TOOLS)

    assert body["message"]["tool_calls"] == PARIS_CALL
    assert body["message"]["content"] == ""


def test_chat_complete_without_tools_returns_the_text(monkeypatch):
    body = _chat(monkeypatch, [JSON_CALL], None)

    assert "tool_calls" not in body["message"]
    assert body["message"]["content"] == JSON_CALL
    assert body["done_reason"] == "stop"


@pytest.mark.parametrize("tokens", [
    ["<tool_call>", JSON_CALL[len("<tool_call>"):]],      # the tag arrives as one token
    ["<tool", JSON_CALL[len("<tool"):]],                  # the tag is split over two tokens
], ids=["tag-one-token", "tag-split"])
def test_json_chat_stream_sends_one_tool_chunk(monkeypatch, tokens):
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True)

    assert len(chunks) == 2, chunks
    assert chunks[0]["message"]["tool_calls"] == PARIS_CALL
    assert chunks[0]["message"]["content"] == ""
    assert chunks[0]["done"] is False
    assert chunks[1]["done"] is True and chunks[1]["done_reason"] == "tool_calls"


def test_chat_stream_text_answer_is_streamed_as_text(monkeypatch):
    tokens = ["The weather ", "is sunny."]
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True)

    assert [c["message"]["content"] for c in chunks[:-1]] == tokens
    assert all("tool_calls" not in c["message"] for c in chunks)
    assert chunks[-1]["done"] is True and chunks[-1]["done_reason"] == "stop"


def test_json_openai_complete_shape(monkeypatch):
    body = _chat(monkeypatch, [JSON_CALL], WEATHER_TOOLS, openai=True)

    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = choice["message"]["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"] == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_json_openai_stream_arguments_are_a_json_string(monkeypatch):
    chunks = _chat(monkeypatch, [JSON_CALL], WEATHER_TOOLS, stream=True, openai=True)

    tool_deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert len(tool_deltas) == 1, chunks
    (call,) = tool_deltas[0]["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function" and call["index"] == 0
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == '{"city": "Paris"}'
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finish[-1] == "tool_calls"
