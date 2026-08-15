"""Test package — and the one place that keeps the suite hermetic.

Once a real endpoint is configured (`.env`, or `VC_LLM_*` exported),
`config.live_llm_available()` becomes true and `engine.build_client` starts
returning a real client. The suite would then call the live model: slow, costly,
and no longer a test of this code. Measured the moment the key arrived — the run
went from 5 seconds to minutes and was spending real tokens.

Dry mode is forced here, before any test module imports `config`, so the tests
measure control flow rather than a provider's mood. A test that wants the live
path builds it explicitly.
"""

from __future__ import annotations

import os

# Must happen before `config` is imported anywhere in the suite.
for _name in (
    "VC_LLM_BASE_URL",
    "VC_LLM_API_KEY",
    "VC_MODEL_CHEAP",
    "VC_MODEL_MID",
    "VC_MODEL_STRONG",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL_NAME",
):
    os.environ[_name] = ""

# `config` reads a `.env` at import time; this stops that too.
os.environ["VC_SKIP_ENV_FILE"] = "1"
os.environ["VC_MCP_DEEPWIKI"] = "0"
os.environ["VC_MCP_OPENCLAW"] = "0"

# State now lives in a directory outside the repository, and `config` creates it
# at import time. Without this the suite would write sessions, transcripts and an
# audit log into the operator's real `~/.vcagent` — and a test that resets a
# session would delete their conversation. The directory is per-run and removed
# when the interpreter exits.
import atexit
import shutil
import tempfile

_STATE = tempfile.mkdtemp(prefix="vcagent-tests-")
os.environ["VC_STATE_DIR"] = _STATE
atexit.register(shutil.rmtree, _STATE, True)
