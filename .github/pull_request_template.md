Closes #<!-- issue number -->

## Summary
<!-- What this PR does and why, in 2-4 sentences. Link the root cause for bugs. -->

## Changes
<!-- One bullet per logical change, with the main files. -->
-

## Testing
**Off-device**
```
<!-- exact command, e.g. pytest test_worker_pipe_guard.py -q -->
```
Result: <!-- e.g. 12 passed -->

**Hardware (rk-sbox)**
<!-- Required when src/, Dockerfile, docker-compose* or pyproject.toml change; otherwise write: N/A – no runtime change -->

| Suite | SHA | Verdict | run_id |
|---|---|---|---|
| unit | | | |
| api-e2e | | | |

## Risk and rollback
<!-- What could break (API compatibility, config defaults, NPU memory); how to revert. -->

## Checklist
- [ ] Linked issue acceptance criteria met
- [ ] rk-sbox run is at the PR head SHA (or N/A)
- [ ] API docs (`documentation/api/`) updated if an endpoint changed
- [ ] No secrets, prompts or private hostnames in code, logs or this description

---
> AI disclosure: <!-- e.g. Implemented with Claude Code (AI); verified on RK3588 via rk-sbox; reviewed by @Ftelf. -->
