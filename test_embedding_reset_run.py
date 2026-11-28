"""
Off-device regression tests for issue #39: a generate whose prompt has as many tokens
as the embedding run right before it killed the model worker.

Measured on RK3588 (RKLLM runtime 1.3.1, rk-sbox play sessions 2026-11-28): the
generate fails ``RKLLM_ASSERT(lctx.inp_out_ids ...)`` and the worker dies (SIGABRT)
when the immediately preceding run on the handle was an embedding (hidden-layer
run) of the same length. The text and the KV cache do not matter; any run of another
length in between prevents it.

Fix under test: the worker puts a short hidden-layer run of another length between
the embedding and such an inference (``_reset_after_embedding``).

Uses the in-process worker harness of test_client_drop.py (fake runtime, no NPU, no
fork). The fake runtime here also models the runtime rule above.
"""
import multiprocessing
import sys

import pytest

from test_client_drop import (  # noqa: F401  (env: fixture of the in-process worker harness)
    READ_TIMEOUT, _assert_complete_answer, _read_answer, _recv, _start_worker, env)

EMBEDDED = 16  # tokens of the embedding in these tests


def _classes():
    return sys.modules["rkllama.api.classes"]


@pytest.fixture
def board(env, monkeypatch):
    """Adds the measured runtime rule and embedding results to the fake runtime of ``env``.

    The tag of a task's input picks a variant: "reject" is a prompt the wrapper rejects
    before rkllm_run (no run on the handle), "novector" an embedding that returns no vector."""
    classes = _classes()
    generate, hidden = classes.RKLLMInferMode.RKLLM_INFER_GENERATE, classes.RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
    callback = sys.modules["rkllama.api.callback"]
    rkllm_class = sys.modules["rkllama.api.rkllm"].RKLLM
    state = {"last_run": None, "asserts": [], "runs": []}
    original_run = rkllm_class.run

    def run(self, inference_mode, model_input_type, model_input, options):
        tokens, tag = model_input
        if tag == "reject":
            return
        n = len(tokens)
        tag = tag or "reset"  # the worker's reset run carries no tag
        state["runs"].append((tag, n))
        last_run, state["last_run"] = state["last_run"], (inference_mode, n)
        if inference_mode == generate and last_run == (hidden, n):
            state["asserts"].append(n)
            raise RuntimeError("RKLLM_ASSERT(lctx.inp_out_ids ...) failed")
        original_run(self, inference_mode, model_input_type, (n, tag), options)
        if inference_mode == hidden:
            # A hidden-layer run reports a hidden state, not text
            callback.global_text.clear()
            callback.global_metrics.clear()
            if tag != "novector":
                callback.last_embeddings.append({"embedding": [0.5], "embd_size": 1, "num_tokens": n})

    monkeypatch.setattr(rkllm_class, "run", run)
    return state


def _task(env, task, mode, n_tokens, tag):
    """Queue a task as WorkerManager does (an inference is followed by its CLEAR_CACHE);
    return the client's end of the request pipe."""
    classes = _classes()
    parent_conn, child_conn = multiprocessing.Pipe()
    env.tasks.put((child_conn, task, mode, classes.RKLLMInputType.RKLLM_INPUT_TOKEN, (list(range(n_tokens)), tag), {}))
    if task == env.worker_mod.WORKER_TASK_INFERENCE:
        env.tasks.put((None, env.worker_mod.WORKER_TASK_CLEAR_CACHE, None, None, None, None))
    return parent_conn


def _embedding(env, n_tokens, tag="e"):
    return _task(env, env.worker_mod.WORKER_TASK_EMBEDDING, _classes().RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER, n_tokens, tag)


def _inference(env, n_tokens, tag):
    return _task(env, env.worker_mod.WORKER_TASK_INFERENCE, _classes().RKLLMInferMode.RKLLM_INFER_GENERATE, n_tokens, tag)


def _run_tasks(env):
    _start_worker(env)


def test_an_inference_of_the_embeddings_length_answers(env, board):
    embedding = _embedding(env, EMBEDDED)
    inference = _inference(env, EMBEDDED, "n")
    _run_tasks(env)

    assert _recv(embedding)["num_tokens"] == EMBEDDED
    answer = _read_answer(env, inference)
    env.worker_thread.join(READ_TIMEOUT)

    assert board["asserts"] == [], "the inference ran right after an embedding of its length"
    _assert_complete_answer(env, answer, "n", EMBEDDED)
    assert board["runs"] == [("e", EMBEDDED), ("reset", 2), ("n", EMBEDDED)], board["runs"]


def test_an_inference_of_another_length_gets_no_reset_run(env, board):
    embedding = _embedding(env, EMBEDDED)
    inference = _inference(env, EMBEDDED + 1, "n")
    _run_tasks(env)

    _recv(embedding)
    answer = _read_answer(env, inference)
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, answer, "n", EMBEDDED + 1)
    assert board["runs"] == [("e", EMBEDDED), ("n", EMBEDDED + 1)], board["runs"]


def test_a_prompt_rejected_before_the_runtime_leaves_the_embedding_last(env, board):
    """The rejected inference never ran on the handle: the next one of the embedding's length still needs the reset."""
    embedding = _embedding(env, EMBEDDED)
    rejected = _inference(env, EMBEDDED + 1, "reject")
    inference = _inference(env, EMBEDDED, "n")
    _run_tasks(env)

    _recv(embedding)
    _read_answer(env, rejected)
    answer = _read_answer(env, inference)
    env.worker_thread.join(READ_TIMEOUT)

    assert board["asserts"] == [], "the inference ran right after an embedding of its length"
    _assert_complete_answer(env, answer, "n", EMBEDDED)


def test_an_embedding_without_a_vector_resets_before_any_inference(env, board):
    """Its length is unknown, so it may equal the inference's."""
    embedding = _embedding(env, EMBEDDED, "novector")  # kept open: a closed reader makes the worker skip the task
    inference = _inference(env, EMBEDDED + 1, "n")
    _run_tasks(env)

    answer = _read_answer(env, inference)
    embedding.close()
    env.worker_thread.join(READ_TIMEOUT)

    _assert_complete_answer(env, answer, "n", EMBEDDED + 1)
    assert board["runs"] == [("novector", EMBEDDED), ("reset", 2), ("n", EMBEDDED + 1)], board["runs"]


@pytest.mark.parametrize("embedding_tokens, input_type, prompt_tokens, expected", [
    (None, "TOKEN", 16, False),       # the last run was not an embedding
    (16, "TOKEN", 16, True),          # the measured crash
    (16, "TOKEN", 17, False),         # another length: safe, no cost
    (-1, "TOKEN", 17, True),          # unknown embedding length
    (16, "MULTIMODAL", None, True),   # the prefill length is not known in the worker
])
def test_needs_reset_after_embedding(env, embedding_tokens, input_type, prompt_tokens, expected):
    worker_mod = env.worker_mod
    input_types = _classes().RKLLMInputType
    model_input_type = getattr(input_types, f"RKLLM_INPUT_{input_type}")
    model_input = (list(range(prompt_tokens)), None) if prompt_tokens is not None else ("prompt", None)
    if embedding_tokens == -1:
        embedding_tokens = worker_mod.EMBEDDING_TOKENS_UNKNOWN
    assert worker_mod._needs_reset_after_embedding(embedding_tokens, model_input_type, model_input) is expected
