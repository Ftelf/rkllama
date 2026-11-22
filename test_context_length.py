"""
Off-device regression tests: the early prompt-length check uses the context
length the model was LOADED with, not the num_ctx of the current request.

rkllm_init fixes max_context_len when a model loads; a later request with a
different num_ctx (Open WebUI sends one when "Context Length" is set for a
model) does not change it. The early check used the request's value, so:
  - a request asking for more context than the model is loaded with passed the
    check and then failed in RKLLM.run inside the started 200 stream (empty
    answer in Open WebUI), and
  - a request asking for less rejected prompts the model would accept.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime); server_utils is loaded with flask/transformers stubbed.
"""
import importlib.util
import sys
import types

import pytest

from test_worker_pipe_guard import SRC, _install_fakes, _load_worker_module


def _model_utils():
    _install_fakes()
    return sys.modules["rkllama.api.model_utils"]


def _server_utils(monkeypatch):
    """Load the REAL server_utils.py with its heavy imports stubbed."""
    _install_fakes()
    flask = types.ModuleType("flask")
    flask.jsonify = lambda *a, **k: (a, k)
    flask.Response = object
    flask.stream_with_context = lambda gen: gen
    monkeypatch.setitem(sys.modules, "flask", flask)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    format_utils = types.ModuleType("rkllama.api.format_utils")
    for name in ("create_format_instruction", "validate_format_response", "get_tool_calls", "handle_ollama_response",
                 "handle_ollama_embedding_response", "get_base64_image_from_pil", "get_url_image_from_pil"):
        setattr(format_utils, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "rkllama.api.format_utils", format_utils)
    variables = types.ModuleType("rkllama.api.variables")
    monkeypatch.setitem(sys.modules, "rkllama.api.variables", variables)
    _load_worker_module()
    spec = importlib.util.spec_from_file_location("rkllama.api.server_utils", str(SRC / "rkllama" / "api" / "server_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_context_length_uses_num_ctx_else_the_configured_default():
    model_utils = _model_utils()
    import rkllama.config
    default = int(rkllama.config.get("model", "default_num_ctx"))

    assert model_utils.context_length_from_options(None) == default
    assert model_utils.context_length_from_options({}) == default
    assert model_utils.context_length_from_options({"num_ctx": "8192"}) == 8192    # Modelfile values are strings
    assert model_utils.context_length_from_options({"num_ctx": "16384.0"}) == 16384
    assert model_utils.context_length_from_options({"num_ctx": 12288}) == 12288


@pytest.mark.parametrize("loaded, requested, n_tokens, rejected", [
    (16384, 4096, 8000, False),   # request asks for less than loaded: the model still takes the prompt
    (4096, 16384, 8000, True),    # request asks for more than loaded: reject now, not inside the stream
    (8192, 8192, 8192, False),    # exactly the loaded context fits
    (8192, 8192, 8193, True),
    (None, 8192, 8000, False),    # model value unknown: fall back to the request's num_ctx
    (None, 8192, 9000, True),
])
def test_prompt_check_uses_the_loaded_context(monkeypatch, loaded, requested, n_tokens, rejected):
    server_utils = _server_utils(monkeypatch)

    error = server_utils.EndpointHandler.prompt_context_error(list(range(n_tokens)), {"num_ctx": str(requested)}, loaded)

    if rejected:
        expected = loaded if loaded is not None else requested
        assert error and f"{n_tokens} tokens" in error and f"{expected} tokens" in error, error
    else:
        assert error is None, error


def test_worker_records_the_context_it_loads_the_model_with(monkeypatch):
    _install_fakes()
    worker_mod = _load_worker_module()
    monkeypatch.setattr(worker_mod, "is_rkllm_model", lambda name: True)
    monkeypatch.setattr(worker_mod, "is_gguf_model", lambda name: False)
    monkeypatch.setattr(worker_mod, "get_model_size", lambda name: 1)

    started = []

    class FakeProcess:
        """Stands in for the worker process (macOS spawns, and a spawned child
        would not see the fake modules): reports a successful model load."""
        def __init__(self, target, args):
            self.args = args

        def start(self):
            started.append(self.args)
            result_queue = self.args[2]
            result_queue.put(worker_mod.WORKER_TASK_FINISHED)

    monkeypatch.setattr(worker_mod, "Process", FakeProcess)

    worker = worker_mod.Worker("fake-model", 1)
    assert worker.worker_model_info.num_ctx is None
    assert worker.create_worker_process([1], "unused", "unused", options={"num_ctx": "12288"})
    assert started, "no worker process was started"
    assert worker.worker_model_info.num_ctx == 12288

    manager = worker_mod.WorkerManager.__new__(worker_mod.WorkerManager)  # no orphan reaping / signal handlers
    manager.workers = {"fake-model": worker}
    assert manager.loaded_context_length("fake-model") == 12288
    assert manager.loaded_context_length("not-loaded") is None


# --- /api/show reports the context the model loads with ------------------------------

@pytest.mark.parametrize("modelfile, expected", [
    ('FROM="m.rkllm"\nNUM_CTX=16384\n', 16384),
    ('FROM="m.rkllm"\nNUM_CTX=8192\nTEMPERATURE=0.7\n', 8192),
    ('FROM="m.rkllm"\nTEMPERATURE=0.7\n', None),   # no NUM_CTX: the configured default
])
def test_model_context_length_reads_the_modelfile(tmp_path, modelfile, expected):
    model_utils = _model_utils()
    import rkllama.config
    (tmp_path / "Qwen3.5-2B-test").mkdir()
    (tmp_path / "Qwen3.5-2B-test" / "Modelfile").write_text(modelfile)

    reported = model_utils.modelfile_context_length("Qwen3.5-2B-test", str(tmp_path))

    assert reported == (expected if expected is not None else int(rkllama.config.get("model", "default_num_ctx")))
