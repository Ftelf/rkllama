# Reading the server log: the life of one request

Every request to an RKLLM model (`/api/chat`, `/api/generate`, `/api/embed` and their `/v1` forms) writes one **start** line and one **end** line at INFO level, with the same short random id (8 hex characters):

```
rkllama.request - INFO - request 3fa9c21e start endpoint=chat api=ollama model=qwen2.5:3b
rkllama.request - INFO - request 3fa9c21e end outcome=finished duration_ms=1840 tokens=57
```

Only categories, numbers and the model name are logged. Prompt text, answer text, options and error texts never are.

## Outcomes

| outcome | meaning |
|---|---|
| `finished` | the worker finished the answer |
| `token limit` | the answer was cut by `max_new_tokens`. The worker reports no finish reason, and on RK3588 (rkllm 1.3.x) the runtime counts one token less than the limit at a cut (limits 6, 8, 12, 16 gave 5, 7, 11, 15), so this is `tokens >= max_new_tokens - 1`; a model that ends by itself on exactly that count looks the same |
| `client gone` | the client closed a streamed answer. Only streams can show it: a non-stream request writes nothing until it is done. `tokens` is what was read so far |
| `aborted` | the worker reported empty stats although this request did not time out. A request's timeout aborts only its own run (issue #37), so this is not expected; an unverified possible cause is a run that reports all-zero stats by itself (e.g. a prompt-cache hit) |
| `timed out` | no worker message within `max_seconds_waiting_worker_response` |
| `worker dead` | the worker process exited or closed the pipe without a result |
| `error` | anything else: the worker reported an error, the model could not be loaded (HTTP 500/503), the prompt was longer than the context (HTTP 400), an exception in the handler |

`tokens=-` means the count is unknown (for example a worker error).

## Following one id

The id is also sent to the worker (as the last element of the task) and used by the worker manager. Grep for it:

```
request 3fa9c21e start endpoint=chat api=ollama model=qwen2.5:3b
request 3fa9c21e model reused model=qwen2.5:3b            (or: loading (not loaded) / loading (reload after a dead worker))
request 3fa9c21e queued model=qwen2.5:3b
request 3fa9c21e run started task=inference model=qwen2.5:3b          <- worker process
Running inference for model qwen2.5:3b...                  <- existing worker lines, no id
The client of model qwen2.5:3b disconnected during the answer ...
Run of model qwen2.5:3b stopped after 93 ms (aborted=True, ...)
request 3fa9c21e run ended outcome=client gone model=qwen2.5:3b
request 3fa9c21e end outcome=client gone duration_ms=2210 tokens=12
```

A worker handles one task at a time, so every worker line between `run started` and `run ended` of an id belongs to that request. If the worker dies, the request also logs `request <id> worker exited exitcode=<n>`.

A request that timed out (or whose client left) while its task was still queued is not run: the worker logs `request <id> run skipped outcome=aborted` (or `outcome=client gone`) instead of `run started` / `run ended`. The worker's `run ended outcome=aborted` means the request's own reader aborted it on a timeout; the request's end line then says `timed out`.

## Model lifecycle

Each unload logs one line with its reason, followed by the existing `Worker <model> stopped.` line:

```
Unloading model qwen2.5:3b (reason: idle)
```

Reasons: `idle` (not used for `max_minutes_loaded_in_memory`), `slot` (all NPU domains taken, least recently used idle model evicted), `memory` (not enough RAM or RKNN memory), `dead worker` (the worker process had died), `shutdown` (server stopping), `api` (unload, delete or an empty `generate` request). A load is logged by the existing `Worker for model <model> created and running...` line.

## Not covered

Requests that go to `.gguf` models (proxied to `llama-server`) and to the RKNN endpoints (image, speech, transcription, translation) have no request lines.
