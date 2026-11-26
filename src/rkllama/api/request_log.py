"""Request lifecycle lines of the server log (issue #29).

One request to an rkllm model (chat, generate, embed) logs exactly one start
line and one end line, with the same short random id:

    request 3fa9c21e start endpoint=chat api=ollama model=qwen2.5:3b
    request 3fa9c21e end outcome=finished duration_ms=1840 tokens=57

The id also goes to the worker (as the optional last element of the task tuple)
and to the worker manager, whose lines for the request carry it.

Log lines only: nothing in here changes what a request does. Only categories,
numbers and the model name are logged: never prompt text, response text,
options or error texts, which can hold user content.

This module imports nothing from the rest of the package, so it loads off-device.
"""
import logging
import re
import secrets
import threading
import time

logger = logging.getLogger("rkllama.request")

# The outcomes of a request, as written in the end line
FINISHED = "finished"
TOKEN_LIMIT = "token limit"      # the answer was cut by max_new_tokens
CLIENT_GONE = "client gone"      # the client closed a streamed answer; only streams can see it
ABORTED = "aborted"              # the worker cut the run (abort flag) without this request asking for it
TIMED_OUT = "timed out"          # no worker answer within max_seconds_waiting_worker_response
WORKER_DEAD = "worker dead"      # the worker process exited or closed the pipe without a result
ERROR = "error"                  # anything else that failed

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def clean(value):
    """A value as one line: the model name is client input and must not forge log lines."""
    return _CONTROL_CHARS.sub("?", str(value))


def hit_token_limit(options, token_count):
    """True if the answer was cut by the request's max_new_tokens.

    Decided here because the worker reports no finish reason. On RK3588 (rkllm runtime
    1.3.x, measured with rk-sbox run 20261126T120151Z) the runtime's token count of a
    cut answer is max_new_tokens - 1: limits of 6, 8, 12 and 16 gave counts of 5, 7, 11
    and 15. So the test is ``token_count >= max_new_tokens - 1``. A model that ends by
    itself on exactly that count cannot be told apart. A limit of 0 or less means "the
    runtime's load-time value", which is not known here, so it claims nothing."""
    try:
        limit = int((options or {}).get("max_new_tokens"))
        return limit > 0 and int(token_count) >= limit - 1
    except (TypeError, ValueError):
        return False


def outcome_of_final_stats(options, prompt_token_count, token_count, prompt_eval, eval_time):
    """The outcome for the final message of a run that was not reported as an error.

    A worker that aborted a run (abort flag) sends all-zero stats (run_rkllm_worker);
    a run that produced a result never has a prompt of 0 tokens."""
    if not any((prompt_token_count, token_count, prompt_eval, eval_time)):
        return ABORTED
    if hit_token_limit(options, token_count):
        return TOKEN_LIMIT
    return FINISHED


class RequestTrace:
    """The start and end line of one request. end() writes at most once, from any thread."""

    def __init__(self, endpoint, model, api="ollama"):
        self.id = secrets.token_hex(4)
        self.endpoint = endpoint
        self.model = model
        self.api = api
        # Tokens of the answer: the worker's count when it finished, else the tokens seen so far
        self.tokens = None
        self.outcome = None
        # Set by a handler that returns a streamed answer: the stream's generator ends the trace
        self.owned_by_stream = False
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._ended = False

    def start(self):
        logger.info("request %s start endpoint=%s api=%s model=%s", self.id, self.endpoint, self.api, clean(self.model))

    def set_outcome(self, outcome):
        """The first outcome set wins: what the parent saw first (timeout, client gone)
        is not overwritten by the worker's message that comes after it."""
        if self.outcome is None:
            self.outcome = outcome

    def end(self, outcome=None):
        if outcome is not None:
            self.set_outcome(outcome)
        with self._lock:
            if self._ended:
                return
            self._ended = True
        logger.info("request %s end outcome=%s duration_ms=%d tokens=%s",
                    self.id, self.outcome or ERROR, (time.monotonic() - self._started) * 1000,
                    "-" if self.tokens is None else self.tokens)


class NullTrace:
    """A trace that logs nothing: for handler methods called without one."""
    id = None
    tokens = None
    outcome = None
    owned_by_stream = False

    def start(self):
        pass

    def set_outcome(self, outcome):
        pass

    def end(self, outcome=None):
        pass


NULL_TRACE = NullTrace()


def traced_stream(trace, stream):
    """Yield the chunks of ``stream`` and end ``trace`` when it is over: finished,
    failed or closed by the server because the client went away."""
    try:
        yield from stream
    except GeneratorExit:
        # Closed from outside while the answer was running (or between two chunks)
        trace.set_outcome(CLIENT_GONE)
        raise
    finally:
        trace.end()
