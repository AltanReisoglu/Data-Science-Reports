"""Compare the two fan-in engines under the same injected failure.

    .venv/bin/python pipeline/compare_fanin.py

Same company, same three analysts, same dry-mode scripts. The only variable is
how results are gathered:

* ``graph.py``  — AgentChat `GraphFlow`, join owned by the framework
* ``fanin.py``  — `autogen_core` pub/sub, results drained into a queue we own

Each is run three ways: clean, with one branch failing behind
`engine.ResilientClient`, and with one branch failing raw. The question is not
which is faster — it is **how many completed branches survive a failure that has
nothing to do with them.**

Follows the `poc/kiyas.py` convention: numbered scenarios, one table, a JSON file.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config          # noqa: E402
import engine          # noqa: E402
import fanin           # noqa: E402
import graph           # noqa: E402
from agents import analysts  # noqa: E402
from schemas import Company, Signal, Source  # noqa: E402
from datetime import datetime, timezone      # noqa: E402

TIMEOUT = 8.0


def sample_company() -> Company:
    return Company(
        name="Acme",
        domain="acme.com",
        description="comparison fixture",
        signals=[
            Signal(
                kind="funding_round",
                summary="Acme raises $12M",
                date=datetime(2026, 8, 1, tzinfo=timezone.utc),
                source=Source(name="hn", url="https://news.ycombinator.com/item?id=1"),
            )
        ],
    )


class Exploding:
    def __init__(self, inner):
        self._inner = inner

    async def create(self, *args, **kwargs):
        raise RuntimeError("branch crashed on purpose")

    def __getattr__(self, item):
        return getattr(self._inner, item)


def patch_team_branch(mode: str):
    """mode: 'clean' | 'wrapped' (failure behind ResilientClient) | 'raw'."""
    original = analysts.build_analysts

    def patched(company, ledger):
        technical, market, team = original(company, ledger)
        if mode == "wrapped":
            team._model_client._inner = Exploding(team._model_client._inner)
        elif mode == "raw":
            team._model_client = Exploding(team._model_client)
        return technical, market, team

    return original, patched


async def run_case(engine_name: str, mode: str) -> dict:
    original, patched = patch_team_branch(mode)
    analysts.build_analysts = patched  # type: ignore[assignment]
    ledger = engine.Ledger()
    module = graph if engine_name == "graphflow" else fanin
    try:
        branches, _score, measurement = await module.enrich(
            sample_company(), ledger, timeout=TIMEOUT
        )
    except Exception as e:
        analysts.build_analysts = original  # type: ignore[assignment]
        await ledger.close()
        return {
            "engine": engine_name, "failure": mode, "survivors": 0,
            "reported": 0, "ms": 0, "note": f"raised: {type(e).__name__}",
        }
    finally:
        analysts.build_analysts = original  # type: ignore[assignment]
        await ledger.close()

    survivors = [b.branch for b in branches if b.succeeded]
    return {
        "engine": engine_name,
        "failure": mode,
        "survivors": len(survivors),
        "reported": len(branches),
        "ms": measurement.sure_ms,
        "note": measurement.durma_nedeni[:60],
    }


async def main() -> None:
    print(f"\nfan-in comparison · dry mode · deadline {TIMEOUT:.0f}s")
    print("the number that matters is 'survivors': completed branches that were kept\n")

    rows = []
    for engine_name in ("graphflow", "fanin"):
        for mode in ("clean", "wrapped", "raw"):
            rows.append(await run_case(engine_name, mode))

    header = f"  {'engine':<11} {'injected failure':<18} {'survivors':>9} {'reported':>9} {'ms':>7}  note"
    print(header)
    print("  " + "-" * (len(header) + 6))
    for row in rows:
        print(
            f"  {row['engine']:<11} {row['failure']:<18} "
            f"{row['survivors']:>9} {row['reported']:>9} {row['ms']:>7}  {row['note']}"
        )

    out = config.OUTPUT / "fanin-comparison.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  json: {out}\n")


if __name__ == "__main__":
    asyncio.run(main())
