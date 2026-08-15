"""Model factory (three tiers) and measurement collection.

**The measurement skeleton was not rewritten** — the ``Olcum`` dataclass from
`poc/motor.py` is imported directly (docs/04 §5), so the POC and the pipeline
report the same metrics under the same definition. The instrument that measured
the 63.7% spread between orchestration patterns is the instrument that measures
production.

Two modes:

* **live** — when ``config.live_llm_available()`` is true, an OpenAI-compatible
  endpoint with a separate model name per tier.
* **dry** — otherwise, ``ReplayChatCompletionClient``: no network, no key,
  deterministic. The whole system runs end to end in this mode, and what gets
  measured is the *control flow* rather than the model's mood.

Dry mode is not a demo trick, it is the discipline inherited from the POC:
*hold constant every variable you are not trying to measure.*
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Literal, Sequence

from autogen_core.models import ChatCompletionClient, ModelInfo, RequestUsage

import config

# poc/ is on the path solely for `Olcum`, so measurement has one definition.
_POC = Path(__file__).resolve().parent.parent / "poc"
if str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))
from motor import Olcum as Measurement  # noqa: E402  (poc/motor.py)

Tier = Literal["cheap", "mid", "strong"]

# Dry mode needs structured_output=True for `output_content_type` to work, and
# function_calling=True for tool calls.
DRY_MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="unknown",
    structured_output=True,
)


# A branch that fails announces itself in its own output instead of raising.
BRANCH_FAILURE_MARKER = "__BRANCH_FAILED__"


class ResilientClient:
    """Wraps a model client so a failing call cannot abort its sibling branches.

    This exists because of a measurement, not a preference. In
    `poc/desen_5_core_aktor.py` a crashing handler was shown to open the join
    barrier early and make sibling results vanish silently. The same shape was
    then reproduced one layer up, in the AgentChat graph: when one branch of the
    enrichment fan-out raises, the team run aborts and the analyses that had
    already completed are lost with it — a run with three branches came back
    holding one.

    Letting the exception escape means losing work that was already paid for.
    So the failure is converted into a message: the branch reports its own
    failure, the join still receives three inputs, and `graph.py` turns the
    marked branch into an entry in ``Score.missing_data``. A silent partial
    result becomes a stated absence of information.
    """

    def __init__(self, inner: ChatCompletionClient) -> None:
        self._inner = inner

    async def create(self, *args, **kwargs):
        from autogen_core.models import CreateResult, RequestUsage

        try:
            return await self._inner.create(*args, **kwargs)
        except Exception as e:
            return CreateResult(
                finish_reason="stop",
                content=f"{BRANCH_FAILURE_MARKER} {type(e).__name__}: {e}",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

    def __getattr__(self, item):
        return getattr(self._inner, item)


class Ledger:
    """Collects every client opened during a run and reduces them to measurements.

    ``measurement()`` reports the **delta since the previous call**, not the
    running total. A ledger lives for a whole scan while measurements are taken
    per company, so cumulative reads would count the same tokens once per
    company and then again when the per-company rows are summed.
    """

    def __init__(self) -> None:
        self.clients: list[ChatCompletionClient] = []
        self._counted = {"prompt": 0, "completion": 0, "calls": 0}

    def client(self, tier: Tier, script: Sequence[str] | None = None) -> ChatCompletionClient:
        c = build_client(tier, script)
        # The raw client is what gets accounted and closed; agents receive the
        # resilient wrapper so one branch's failure cannot take its siblings down.
        self.clients.append(c)
        return ResilientClient(c)  # type: ignore[return-value]

    def raw_client(self, tier: Tier, script: Sequence[str] | None = None) -> ChatCompletionClient:
        """A client whose failures are allowed to raise.

        `ResilientClient` turns a failed call into text so one branch of a
        fan-out cannot abort its siblings. In a conversation there are no
        siblings to protect, and swallowing the failure would hand the reader a
        crash message dressed as an answer — an endpoint outage would read as
        the analyst's opinion. Accounted and closed like any other client.
        """
        c = build_client(tier, script)
        self.clients.append(c)
        return c

    def _totals(self) -> dict[str, int]:
        totals = {"prompt": 0, "completion": 0, "calls": 0}
        for c in self.clients:
            usage: RequestUsage = c.total_usage()
            totals["prompt"] += usage.prompt_tokens
            totals["completion"] += usage.completion_tokens
            totals["calls"] += len(getattr(c, "create_calls", []) or [])
        return totals

    def measurement(self, label: str) -> Measurement:
        totals = self._totals()
        m = Measurement(desen=label, mod="live" if config.live_llm_available() else "dry")
        m.prompt_token = totals["prompt"] - self._counted["prompt"]
        m.completion_token = totals["completion"] - self._counted["completion"]
        m.llm_cagrisi = totals["calls"] - self._counted["calls"]
        self._counted = totals
        return m

    def uncounted(self, label: str) -> Measurement:
        """Whatever has not been attributed to a company yet (triage, memos)."""
        return self.measurement(label)

    async def close(self) -> None:
        for c in self.clients:
            try:
                await c.close()
            except Exception:
                pass
        self.clients.clear()


def build_client(tier: Tier, script: Sequence[str] | None = None) -> ChatCompletionClient:
    """Return a model client for the given tier.

    ``script`` is only consulted in dry mode: the replies that agent will hand
    back, in order.
    """
    if config.live_llm_available():
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model = config.MODEL_TIERS[tier]
        try:
            # A known OpenAI model carries its own capability record, and letting
            # the client find it is better than overriding it with a guess.
            return OpenAIChatCompletionClient(
                model=model, api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                timeout=config.LLM_TIMEOUT, max_retries=config.LLM_MAX_RETRIES,
            )
        except ValueError:
            # "model_info is required when model name is not a valid OpenAI model".
            # Any OpenAI-*compatible* endpoint lands here — which is the whole
            # point of `VC_LLM_BASE_URL` — so the capability record has to be
            # supplied. `config.LIVE_MODEL_INFO` is overridable per deployment
            # because getting it wrong is silent: claim structured output the
            # model lacks and the scorer fails at the last step of the funnel.
            return OpenAIChatCompletionClient(
                model=model,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
                model_info=config.LIVE_MODEL_INFO,
                timeout=config.LLM_TIMEOUT,
                max_retries=config.LLM_MAX_RETRIES,
            )

    from autogen_ext.models.replay import ReplayChatCompletionClient

    return ReplayChatCompletionClient(list(script or ["…"]), model_info=DRY_MODEL_INFO)


def combine(measurements: Iterable[Measurement], label: str = "total") -> Measurement:
    """Reduce per-company measurements to a single row."""
    total = Measurement(desen=label, mod="live" if config.live_llm_available() else "dry")
    for m in measurements:
        total.sure_ms += m.sure_ms
        total.mesaj_sayisi += m.mesaj_sayisi
        total.llm_cagrisi += m.llm_cagrisi
        total.arac_cagrisi += m.arac_cagrisi
        total.prompt_token += m.prompt_token
        total.completion_token += m.completion_token
    return total
