---
name: Bug
about: Something in rkllama is broken or behaves wrongly
title: "<symptom, e.g. Last token leaks into the next response>"
---

## Summary
<!-- 2-3 sentences: what goes wrong, who is affected, how bad it is. -->

## Environment
- rkllama ref: <!-- full SHA or release -->
- Board / SoC: <!-- e.g. Orange Pi 5 Max, RK3588 (rk-sbox generation G1) -->
- Runtime: <!-- native / Docker image tag; librkllmrt / librknnrt version -->
- Model(s): <!-- name, format (.rkllm/.gguf), context -->
- Client / endpoint: <!-- e.g. Open WebUI, POST /api/chat, stream=true -->

## Steps to reproduce
1.
2.

## Expected behaviour

## Actual behaviour

## Evidence
<!-- Logs, rk-sbox run_id + verdict, minimal output. Redact prompts, hostnames, tokens. -->

## Suspected cause
<!-- Optional. file:line if known; write "unverified" if it is a hypothesis. -->

## Acceptance criteria
- [ ] <!-- observable condition that proves the bug is fixed -->
- [ ] Regression test added (or reason why it cannot run off-device)

---
> AI disclosure: <!-- e.g. Drafted with Claude Code (AI); evidence from rk-sbox; reviewed by @Ftelf. -->
