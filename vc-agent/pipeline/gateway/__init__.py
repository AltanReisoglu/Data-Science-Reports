"""The gateway: one process that owns sessions, routing, hooks and the audit.

The shape is OpenClaw's (docs/13), the engine is AutoGen. That split is the whole
idea. OpenClaw solved the operational half of running an agent — who is talking,
which conversation this belongs to, when it resets, what gets recorded, what is
allowed to leave the machine — and solved it in the open. None of that is
AutoGen's problem, and AutoGen does not answer it.

What we take:

* **Gateway as single source of truth.** One long-lived process owns sessions and
  routing. Not a rule about tidiness: two processes writing one session index is
  a corrupted index.
* **Sessions keyed by origin** (`sessions.py`), with the same lifecycle OpenClaw
  uses — daily reset, idle reset, manual reset, pruning.
* **Hooks as the extension surface** (`hooks.py`), with OpenClaw's decision rules:
  `before_tool_call → block` is terminal, a crashing hook is quarantined rather
  than allowed to silence the agent.
* **Audit that stores metadata, not content.**

What we deliberately do not take, and why, is in `docs/15`.
"""
