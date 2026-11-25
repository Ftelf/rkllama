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
import logging
import re
import sys
import time
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


def _chat(monkeypatch, tokens, tools, stream=False, openai=False, enable_thinking=False):
    """Run one /api/chat (or /v1/chat/completions) request through ChatEndpointHandler.
    Non-stream: the response body as a dict. Stream: the list of chunks (dicts for
    Ollama NDJSON lines, dicts of the `data:` payloads for OpenAI SSE)."""
    server_utils = _server_utils(monkeypatch, tokens)
    result = server_utils.ChatEndpointHandler.handle_request(
        "qwen-test", [{"role": "user", "content": "q"}], stream=stream, tools=tools, is_openai_request=openai,
        enable_thinking=enable_thinking)
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


# --- XML form (Qwen3-Coder / Qwen3.5) ---------------------------------------------------

def _xml_call(name, *params, before="\n", after="\n"):
    """<tool_call> block in the XML form; params are (name, raw value) pairs, values as written."""
    body = "".join(f"<parameter={p}>{v}</parameter>\n" for p, v in params)
    return f"<tool_call>{before}<function={name}>\n{body}</function>{after}</tool_call>"


XML_CALL = _xml_call("get_weather", ("city", "\nParis\n"))   # the answer seen on hardware (rk-sbox F-015)

FORECAST_TOOLS = [{"type": "function", "function": {
    "name": "get_forecast", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}}}}]

NOTE_TOOLS = [{"type": "function", "function": {
    "name": "add_note", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "urgent": {"type": "boolean"}, "due": {"type": ["string", "null"]}}}}}]
NOTE_CALL = _xml_call("add_note", ("text", "\nDon't forget the milk\n"), ("urgent", "\ntrue\n"), ("due", "\nnull\n"))
NOTE_ARGUMENTS = {"text": "Don't forget the milk", "urgent": True, "due": None}


def _same(actual, expected):
    """Equal and of the same type (42 == 42.0 == True in Python)."""
    return type(actual) is type(expected) and actual == expected


def test_xml_reproduction_answer_is_a_tool_call(fu):
    calls, content, used_xml = fu.extract_tool_calls(XML_CALL, WEATHER_TOOLS)

    assert calls == PARIS_CALL   # exactly one leading and one trailing newline removed, not "\nParis\n"
    assert content == ""
    assert used_xml is True
    assert fu.get_tool_calls(XML_CALL, WEATHER_TOOLS) == PARIS_CALL


@pytest.mark.parametrize("raw, schema, expected", [
    ("\n42\n", {"type": "integer"}, 42),
    ("\n-7\n", {"type": "integer"}, -7),
    ("\n 12 \n", {"type": "integer"}, 12),                  # non-strings are stripped fully
    ("\n2.5\n", {"type": "number"}, 2.5),
    ("\n3\n", {"type": "number"}, 3),                       # integral literal stays an int
    ("\n-1.5e3\n", {"type": "number"}, -1500.0),
    ("\nTrue\n", {"type": "boolean"}, True),                # case-insensitive
    ("\nfalse\n", {"type": "boolean"}, False),
    ("\nnull\n", {"type": "null"}, None),
    ('\n{"unit": "C", "hours": [1, 2]}\n', {"type": "object"}, {"unit": "C", "hours": [1, 2]}),
    ('\n[1, "x", null]\n', {"type": "array"}, [1, "x", None]),
    ("\n42\n", {"type": "string"}, "42"),                   # a string schema keeps the text
    ("\n42\n", {"type": ["integer", "null"]}, 42),
    ("\nnull\n", {"type": ["integer", "null"]}, None),
    ("\nnull\n", {"type": ["string", "null"]}, None),       # null is tried first
    ("\n42\n", {"type": ["string", "integer"]}, "42"),      # then the listed order
    ("\n42\n", {"type": ["integer", "string"]}, 42),
    ("\nabc\n", {"type": ["integer", "string"]}, "abc"),
    ("\n5\n", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, 5),
    ("\nfive\n", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "five"),
    ("\nnull\n", {"oneOf": [{"type": "boolean"}, {"type": "null"}]}, None),
    ("\ntrue\n", {"oneOf": [{"type": "boolean"}, {"type": "null"}]}, True),
])
def test_xml_value_types_follow_the_schema(fu, raw, schema, expected):
    assert _same(fu.coerce_xml_value(raw, schema), expected)


@pytest.mark.parametrize("raw, schema, expected", [
    ("\nnan\n", {"type": "number"}, "nan"),
    ("\ninf\n", {"type": "number"}, "inf"),
    ("\n1e999\n", {"type": "number"}, "1e999"),             # overflows to inf
    ("\n3.0x\n", {"type": "number"}, "3.0x"),
    ("\n4.5\n", {"type": "integer"}, "4.5"),
    ("\nyes\n", {"type": "boolean"}, "yes"),
    ("\nNone\n", {"type": "null"}, "None"),
    ("\n{bad\n", {"type": "object"}, "{bad"),
    ("\n[1]\n", {"type": "object"}, "[1]"),
    ('\n{"a": 1}\n', {"type": "array"}, '{"a": 1}'),
    ('\n{"a": NaN}\n', {"type": "object"}, '{"a": NaN}'),
    ('\n{"a": 1e999}\n', {"type": "object"}, '{"a": 1e999}'),
    ("[" * 100000, {"type": "array"}, "[" * 100000),       # too deep for json.loads
    ("\nnan\n", {"type": ["number", "null"]}, "nan"),
    ("\nmaybe\n", {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "maybe"),
])
def test_xml_value_that_does_not_convert_stays_a_string(fu, raw, schema, expected):
    assert _same(fu.coerce_xml_value(raw, schema), expected)


@pytest.mark.parametrize("schema", [{"type": "integer"}, {"type": "number"}])
def test_xml_integer_longer_than_int_accepts_stays_a_string(fu, schema):
    # The int() digit limit exists only from Python 3.9.14 / 3.10.7 / 3.11 on, and 0 turns it off.
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not limit:
        pytest.skip("this Python has no int() digit limit")
    digits = "9" * (limit + 1)

    assert _same(fu.coerce_xml_value(f"\n{digits}\n", schema), digits)


@pytest.mark.parametrize("schema", [None, {}, {"type": "weird"}, {"$ref": "#/$defs/Days"}, {"anyOf": [{"$ref": "#/x"}]}])
def test_xml_value_without_a_usable_schema_is_a_string(fu, schema):
    assert _same(fu.coerce_xml_value("\n42\n", schema), "42")


@pytest.mark.parametrize("tools", [
    None,                                                                   # request without tools
    WEATHER_TOOLS,                                                          # tool not in the request
    [{"type": "function", "function": {"name": "get_forecast"}}],           # tool without parameters
], ids=["no-tools", "unknown-tool", "no-schema"])
def test_xml_values_are_strings_without_a_schema_for_the_tool(fu, tools):
    answer = _xml_call("get_forecast", ("days", "\n3\n"), ("metric", "\ntrue\n"))

    calls, _, _ = fu.extract_tool_calls(answer, tools)

    assert calls == [{"function": {"name": "get_forecast", "arguments": {"days": "3", "metric": "true"}}}]


def test_xml_values_are_typed_per_parameter_of_the_called_tool(fu):
    tools = WEATHER_TOOLS + FORECAST_TOOLS
    answer = _xml_call("get_forecast", ("city", "\n42\n"), ("days", "\n3\n"), ("metric", "\nfalse\n"), ("extra", "\n7\n"))

    (call,) = fu.get_tool_calls(answer, tools)

    arguments = call["function"]["arguments"]
    assert _same(arguments["city"], "42") and _same(arguments["days"], 3) and _same(arguments["metric"], False)
    assert _same(arguments["extra"], "7")                      # parameter missing from the schema


@pytest.mark.parametrize("raw, expected", [
    ("Paris", "Paris"),
    ("\n\nParis\n\n", "\nParis\n"),                            # exactly one newline each side
    ("\n  two\nlines <b>a</b> x > y\n", "  two\nlines <b>a</b> x > y"),
    ("\n  leading spaces kept  \n", "  leading spaces kept  "),
    ("\n", ""),
    ("\r\nParis\r\n", "Paris"),                                # Windows line breaks count as one newline
    ("\r\n\r\nParis\r\n\r\n", "\r\nParis\r\n"),
    ("\r\nParis\n", "Paris"),
    ("\r\n", ""),
    ("\r\n  two\r\nlines\r\n", "  two\r\nlines"),               # inner line breaks are kept as written
])
def test_xml_string_values_keep_their_text(fu, raw, expected):
    (call,) = fu.get_tool_calls(_xml_call("get_weather", ("city", raw)), WEATHER_TOOLS)

    assert call["function"]["arguments"] == {"city": expected}


def test_xml_names_are_stripped_and_order_is_kept(fu):
    answer = "<tool_call>\n<function= get_forecast >\n<parameter= metric >\nfalse\n</parameter>\n" \
             "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"

    (call,) = fu.get_tool_calls(answer, FORECAST_TOOLS)

    assert call["function"]["name"] == "get_forecast"
    assert list(call["function"]["arguments"].items()) == [("metric", False), ("city", "Oslo")]


def test_xml_duplicate_empty_and_no_parameters(fu):
    duplicate = _xml_call("get_weather", ("city", "\nParis\n"), ("city", "\nLyon\n"))
    empty = _xml_call("get_weather", ("city", ""))
    none = _xml_call("get_time")
    tight = "<tool_call><function=get_time></function></tool_call>"

    assert fu.get_tool_calls(duplicate) == [{"function": {"name": "get_weather", "arguments": {"city": "Lyon"}}}]
    assert fu.get_tool_calls(empty) == [{"function": {"name": "get_weather", "arguments": {"city": ""}}}]
    assert fu.get_tool_calls(none) == [{"function": {"name": "get_time", "arguments": {}}}]
    assert fu.get_tool_calls(tight) == [{"function": {"name": "get_time", "arguments": {}}}]


def test_several_blocks_give_several_calls_in_text_order(fu):
    json_call = '<tool_call>\n{"name": "get_time", "arguments": {"zone": "CET"}}\n</tool_call>'
    second_xml = _xml_call("get_weather", ("city", "\nLyon\n"))
    answer = f"Checking both.\n{XML_CALL}\nand\n{second_xml}\nthen\n{json_call}\nDone."

    calls, content, used_xml = fu.extract_tool_calls(answer, WEATHER_TOOLS)

    assert calls == [PARIS_CALL[0],
                     {"function": {"name": "get_weather", "arguments": {"city": "Lyon"}}},
                     {"function": {"name": "get_time", "arguments": {"zone": "CET"}}}]
    assert content == "Checking both.\n\nand\n\nthen\n\nDone."
    assert used_xml is True
    assert fu.get_tool_calls(json_call + XML_CALL) == [calls[2], calls[0]]


# Malformed XML blocks, each with a parameter value that looks like a JSON call: the block
# must stay text, and the value must not become a call (the JSON code would find it).
LOOKS_LIKE_JSON = '\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
MALFORMED = {
    "unclosed-tool_call": ("<tool_call>\n<function=get_weather>\n<parameter=city>" + LOOKS_LIKE_JSON
                           + "</parameter>\n</function>\n", None),
    "empty-function-name": (_xml_call("", ("city", LOOKS_LIKE_JSON)), "empty or invalid function name"),
    "blank-function-name": (_xml_call("  ", ("city", LOOKS_LIKE_JSON)), "empty or invalid function name"),
    "newline-in-function-name": (_xml_call("get\nweather", ("city", LOOKS_LIKE_JSON)), "empty or invalid function name"),
    "empty-parameter-name": (_xml_call("get_weather", ("", LOOKS_LIKE_JSON)), "empty or invalid parameter name"),
    "missing-</function>": ("<tool_call>\n<function=get_weather>\n<parameter=city>" + LOOKS_LIKE_JSON
                            + "</parameter>\n</tool_call>", "text between elements or missing </function>"),
    "missing-</parameter>": ("<tool_call>\n<function=get_weather>\n<parameter=city>" + LOOKS_LIKE_JSON
                             + "</function>\n</tool_call>", "missing </parameter>"),
    "text-between-elements": ("<tool_call>\n<function=get_weather>\nI will look it up.\n<parameter=city>"
                              + LOOKS_LIKE_JSON + "</parameter>\n</function>\n</tool_call>",
                              "text between elements or missing </function>"),
    "text-after-</function>": (_xml_call("get_weather", ("city", LOOKS_LIKE_JSON), after="\nthanks\n"),
                               "text after </function>"),
    "two-functions": ("<tool_call>\n<function=get_weather>\n<parameter=city>" + LOOKS_LIKE_JSON
                      + "</parameter>\n</function>\n<function=get_time>\n</function>\n</tool_call>",
                      "text after </function>"),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_xml_block_stays_text(fu, caplog, case):
    answer, reason = MALFORMED[case]
    caplog.set_level(logging.INFO, logger="rkllama.format_utils")

    calls, content, used_xml = fu.extract_tool_calls(answer, WEATHER_TOOLS)

    assert calls == [] and used_xml is False
    assert content == answer.strip()
    rejected = [r.getMessage() for r in caplog.records if "XML tool call" in r.getMessage()]
    if reason is None:   # no complete block: the XML parser is not reached
        assert rejected == []
    else:                # rejected by the grammar for this reason, not by an exception
        assert rejected == [f"XML tool call block not parsed: {reason}; the block stays text"]
    assert all('"' not in m and "Paris" not in m for m in rejected)   # reasons only, no answer text


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_xml_block_chat_complete_is_text(monkeypatch, case):
    answer, _ = MALFORMED[case]

    body = _chat(monkeypatch, [answer], WEATHER_TOOLS)

    assert "tool_calls" not in body["message"]
    assert body["message"]["content"] == answer
    assert body["done_reason"] == "stop"


def test_xml_parameter_that_looks_like_a_json_call_is_only_a_value(fu):
    answer = _xml_call("get_weather", ("city", '\n{"name": "x", "arguments": {}}\n'))

    assert fu.get_tool_calls(answer, WEATHER_TOOLS) == [
        {"function": {"name": "get_weather", "arguments": {"city": '{"name": "x", "arguments": {}}'}}}]


def test_inline_function_tag_in_prose_is_not_a_call(fu):
    answer = f"Write <function=delete_all><parameter=x>1</parameter></function> to call it.\n{XML_CALL}"

    calls, content, _ = fu.extract_tool_calls(answer, WEATHER_TOOLS)

    assert calls == PARIS_CALL
    assert content == "Write <function=delete_all><parameter=x>1</parameter></function> to call it."


def test_xml_parser_exception_keeps_the_block_as_text(fu, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("bug: secret answer text")
    monkeypatch.setattr(fu, "_parse_xml_tool_call_strict", broken)

    calls, content, _ = fu.extract_tool_calls(XML_CALL, WEATHER_TOOLS)

    assert calls == [] and content == XML_CALL
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "XML tool call parser failed (RuntimeError); the block stays text"]


# --- XML form through the chat handler --------------------------------------------------

def test_xml_chat_complete_reproduction(monkeypatch):
    body = _chat(monkeypatch, ["<tool_call>", XML_CALL[len("<tool_call>"):]], WEATHER_TOOLS)

    assert body["message"]["tool_calls"] == PARIS_CALL
    assert body["message"]["content"] == ""
    assert body["done_reason"] == "tool_calls"


def test_xml_chat_complete_types_arguments_from_the_request_tools(monkeypatch):
    answer = _xml_call("get_forecast", ("city", "\nOslo\n"), ("days", "\n3\n"), ("metric", "\ntrue\n"))

    body = _chat(monkeypatch, [answer], FORECAST_TOOLS)

    (call,) = body["message"]["tool_calls"]
    assert call["function"]["arguments"] == {"city": "Oslo", "days": 3, "metric": True}
    assert _same(call["function"]["arguments"]["days"], 3)


def test_xml_chat_complete_keeps_the_text_around_the_blocks(monkeypatch):
    body = _chat(monkeypatch, ["Let me check.\n", XML_CALL, "\nOne moment. "], WEATHER_TOOLS)

    assert body["message"]["tool_calls"] == PARIS_CALL
    assert body["message"]["content"] == "Let me check.\n\nOne moment."
    assert body["done_reason"] == "tool_calls"


@pytest.mark.parametrize("tokens", [
    ["<tool_call>", XML_CALL[len("<tool_call>"):]],
    ["<tool", XML_CALL[len("<tool"):]],
], ids=["tag-one-token", "tag-split"])
def test_xml_chat_stream_sends_one_tool_chunk(monkeypatch, tokens):
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True)

    assert len(chunks) == 2, chunks
    assert chunks[0]["message"]["tool_calls"] == PARIS_CALL
    assert chunks[0]["message"]["content"] == ""
    assert chunks[1]["done"] is True and chunks[1]["done_reason"] == "tool_calls"


def test_xml_chat_stream_types_arguments_from_the_request_tools(monkeypatch):
    answer = _xml_call("get_forecast", ("days", "\n3\n"))

    chunks = _chat(monkeypatch, ["<tool_call>", answer[len("<tool_call>"):]], FORECAST_TOOLS, stream=True)

    assert chunks[0]["message"]["tool_calls"] == [{"function": {"name": "get_forecast", "arguments": {"days": 3}}}]


@pytest.mark.parametrize("filler_tokens", [0, 150], ids=["short", "over-100-tokens"])
def test_stream_with_unparsable_tool_call_block_is_sent_as_text(monkeypatch, filler_tokens):
    tokens = ["<tool_call>", "\n<function=get_weather>\nPlease wait.\n"] + ["word "] * filler_tokens + ["</function>\n</tool_call>"]

    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True)

    assert all("tool_calls" not in c["message"] for c in chunks), chunks[:2]
    assert "".join(c["message"]["content"] for c in chunks) == "".join(tokens)
    assert chunks[-1]["done"] is True and chunks[-1]["done_reason"] == "stop"


# --- OpenAI shape -------------------------------------------------------------------------

def test_xml_openai_complete_shape(monkeypatch):
    body = _chat(monkeypatch, [XML_CALL], WEATHER_TOOLS, openai=True)

    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = choice["message"]["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"] == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_xml_openai_stream_arguments_are_valid_json(monkeypatch):
    ollama_chunks = _chat(monkeypatch, [NOTE_CALL], NOTE_TOOLS, stream=True)
    openai_chunks = _chat(monkeypatch, [NOTE_CALL], NOTE_TOOLS, stream=True, openai=True)

    assert ollama_chunks[0]["message"]["tool_calls"][0]["function"]["arguments"] == NOTE_ARGUMENTS
    tool_deltas = [c["choices"][0]["delta"] for c in openai_chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    (call,) = tool_deltas[0]["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function" and call["index"] == 0
    assert call["function"]["name"] == "add_note"
    assert json.loads(call["function"]["arguments"]) == NOTE_ARGUMENTS
    assert [c["choices"][0]["finish_reason"] for c in openai_chunks if c["choices"]][-1] == "tool_calls"


def _openai_stream_arguments(fu, arguments):
    line = json.dumps({"model": "m", "message": {"role": "assistant", "content": "",
                                                  "tool_calls": [{"function": {"name": "f", "arguments": arguments}}]},
                       "done": False})
    (sse,) = list(fu.ollama_chat_stream_to_openai_chat_completions_chunks([line]))
    chunk = json.loads(sse[len("data: "):])
    return chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize("arguments", [
    {"city": "Paris"},
    {"city": "Zürich", "days": 3, "ratio": 0.5, "nested": {"list": ["a", 1]}},
    {},
])
def test_openai_stream_arguments_bytes_unchanged_for_strings_and_numbers(fu, arguments):
    old_formula = str(arguments).replace("'", '"')

    assert _openai_stream_arguments(fu, arguments) == old_formula


@pytest.mark.parametrize("arguments", [
    {"urgent": True, "due": None},
    {"text": "Don't forget"},
    {"quote": 'say "hi"', "path": "C:\\tmp"},
])
def test_openai_stream_arguments_are_json_for_any_value(fu, arguments):
    assert json.loads(_openai_stream_arguments(fu, arguments)) == arguments


def test_openai_stream_arguments_already_a_string_pass_through(fu):
    assert _openai_stream_arguments(fu, '{"text": "it\'s"}') == '{"text": "it\'s"}'


# --- scan cost --------------------------------------------------------------------------

def test_unclosed_tool_call_tags_are_scanned_in_linear_time(fu):
    """A runaway answer full of <tool_call> tags without </tool_call> must not stall the
    request thread. Each unclosed tag makes `<tool_call>(.*?)</tool_call>` scan to the end
    of the text, so the scan is quadratic: about 3.5 s CPU for 10 000 tags (110 000 chars)
    on a laptop, 4x per doubling; a linear scan takes milliseconds.

    A time bound, because there is no implementation-neutral counter of scan work. CPU
    time (not wall time) keeps scheduler noise out. Margins on a laptop: the quadratic scan
    (before #32) took about 3.4 s, 14x above the bound; the linear scan takes about 0.01 ms,
    far below it. A slower board widens both margins. The block boundaries themselves are
    pinned by test_block_scan_finds_the_same_blocks_as_the_lazy_regex."""
    answer = "<tool_call>" * 10_000

    start = time.process_time()
    calls, content, used_xml = fu.extract_tool_calls(answer, WEATHER_TOOLS)
    elapsed = time.process_time() - start

    assert (calls, content, used_xml) == ([], answer, False)   # behaviour must not change
    assert elapsed < 0.25, f"scanning {len(answer)} chars took {elapsed:.2f} s CPU"


# --- block boundaries -------------------------------------------------------------------

_REFERENCE_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)   # the scan before #32


@pytest.mark.parametrize("text", [
    "",
    "no tags at all",
    "<tool_call></tool_call>",
    "<tool_call>x</tool_call>",
    "a<tool_call>x</tool_call>b<tool_call>y</tool_call>c",
    "<tool_call>\nmulti\nline\n</tool_call>",
    "<tool_call>a<tool_call>b</tool_call>",                  # unclosed tag, then a closed one
    "<tool_call><tool_call></tool_call></tool_call>",
    "<tool_call>x</tool_call><tool_call>",                   # closed block, then an unclosed tag
    "<tool_call>x</tool_call>" + "<tool_call>" * 3,
    "</tool_call><tool_call>x</tool_call>",                  # stray closing tag first
    "<tool_call>x</tool_call></tool_call>",
    "<tool_call>" * 3,
    "<tool_call" + "x</tool_call>",                          # broken opening tag
])
def test_block_scan_finds_the_same_blocks_as_the_lazy_regex(fu, text):
    expected = [(m.start(), m.end(), m.group(1)) for m in _REFERENCE_BLOCK_RE.finditer(text)]
    assert list(fu._tool_call_blocks(text)) == expected


def test_closed_call_after_an_unclosed_tag_is_found(fu):
    # The unclosed tag opens the block, so its text is part of the block and leaves with the call
    answer = "Let me check. <tool_call>oops " + JSON_CALL + " Done."
    calls, content, used_xml = fu.extract_tool_calls(answer, WEATHER_TOOLS)
    assert (calls, content, used_xml) == (PARIS_CALL, "Let me check.  Done.", False)


def test_closed_call_followed_by_unclosed_tags_is_found(fu):
    answer = JSON_CALL + "\n<tool_call>" * 3
    calls, content, used_xml = fu.extract_tool_calls(answer, WEATHER_TOOLS)
    assert (calls, content, used_xml) == (PARIS_CALL, "<tool_call>\n<tool_call>\n<tool_call>", False)


# --- thinking text in tool-call answers (#33) ------------------------------------------
# The stream path keeps thinking text out of the call search (final_response_tokens); the
# non-stream path searches and returns the whole text, thinking included.

THINKING = "<think>\nThe user wants the weather in Paris.\n</think>\n\n"


def test_xml_chat_complete_keeps_thinking_out_of_content(monkeypatch):
    body = _chat(monkeypatch, [THINKING, "I will check.\n", XML_CALL], WEATHER_TOOLS, enable_thinking=True)

    assert body["message"]["tool_calls"] == PARIS_CALL
    content = body["message"]["content"]
    assert "<think>" not in content and "The user wants" not in content, content


@pytest.mark.parametrize("call", [JSON_CALL, XML_CALL], ids=["json", "xml"])
def test_chat_complete_ignores_a_tool_call_inside_thinking(monkeypatch, call):
    tokens = ["<think>\nI could call ", call, " but I know it.\n</think>\n\n", "It is sunny in Paris."]

    body = _chat(monkeypatch, tokens, WEATHER_TOOLS, enable_thinking=True)
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True, enable_thinking=True)

    # the stream path already gives no call; the non-stream answer must agree with it
    assert all("tool_calls" not in c["message"] for c in chunks) and chunks[-1]["done_reason"] == "stop"
    assert "tool_calls" not in body["message"], body["message"]
    assert body["done_reason"] == "stop"


def _stream_result(chunks):
    """(tool_calls, done_reason) of a streamed Ollama chat answer."""
    calls = [c["message"]["tool_calls"] for c in chunks if "tool_calls" in c.get("message", {})]
    return (calls[0] if calls else None), chunks[-1]["done_reason"]


@pytest.mark.parametrize("call", [JSON_CALL, XML_CALL], ids=["json", "xml"])
def test_call_after_thinking_gives_the_same_result_stream_and_non_stream(monkeypatch, call):
    # </think> as its own token, as models send it (the stream path checks the first token
    # only for <think>, so a first token holding the whole block would stay "thinking")
    tokens = ["<think>\nThe user wants the weather in Paris.\n", "</think>", "\n\nI will check.\n", call]

    body = _chat(monkeypatch, tokens, WEATHER_TOOLS, enable_thinking=True)
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True, enable_thinking=True)

    assert (body["message"]["tool_calls"], body["done_reason"]) == (PARIS_CALL, "tool_calls")
    assert _stream_result(chunks) == (PARIS_CALL, "tool_calls")
    assert "The user wants" not in body["message"]["content"]


@pytest.mark.parametrize("call", [JSON_CALL, XML_CALL], ids=["json", "xml"])
def test_thinking_without_end_tag_gives_no_call(monkeypatch, call):
    # Thinking on and no </think>: the whole answer is thinking (the model ran out of tokens
    # while thinking), so a call in it is not a call; the stream path already does this
    tokens = ["<think>\nMaybe ", call, " later."]

    body = _chat(monkeypatch, tokens, WEATHER_TOOLS, enable_thinking=True)
    chunks = _chat(monkeypatch, tokens, WEATHER_TOOLS, stream=True, enable_thinking=True)

    assert ("tool_calls" not in body["message"], body["done_reason"]) == (True, "stop")
    assert _stream_result(chunks) == (None, "stop")


def test_thinking_answer_without_a_call_keeps_its_content(monkeypatch):
    # Out of scope for #33: an answer with tools but no call keeps the thinking in content
    tokens = [THINKING, "It is sunny in Paris."]

    body = _chat(monkeypatch, tokens, WEATHER_TOOLS, enable_thinking=True)

    assert "tool_calls" not in body["message"]
    assert body["message"]["content"] == THINKING + "It is sunny in Paris."


def test_call_with_thinking_off_is_still_found_after_a_think_block(monkeypatch):
    # Thinking off: the answer is searched as a whole, as before #33
    body = _chat(monkeypatch, [THINKING, JSON_CALL], WEATHER_TOOLS)

    assert (body["message"]["tool_calls"], body["done_reason"]) == (PARIS_CALL, "tool_calls")
