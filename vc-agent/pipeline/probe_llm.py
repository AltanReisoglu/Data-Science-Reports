"""Measure what an OpenAI-compatible endpoint actually supports, then print the config.

    .venv/bin/python pipeline/probe_llm.py

`config.LIVE_MODEL_INFO` is a **declaration**, not a measurement — and AutoGen
takes it at its word. Claim a capability the model lacks and nothing fails at
startup; it fails at the far end of the funnel, in the scorer, after the tokens
are already spent. So the capability record should come from asking the endpoint,
which is what this does.

Six checks, each one a thing the pipeline depends on:

1. **reachable** — `/models`, and whether the key is accepted
2. **chat** — a plain completion round-trips
3. **tool calling** — the model returns `tool_calls` when given a tool. The whole
   agent layer rests on this; without it the live conversation cannot call
   `scan_facts`, and enrichment analysts cannot use their tools.
4. **structured output** — `response_format` with a JSON schema. The scorer and
   the memo writer use `output_content_type`, which compiles down to this.
5. **streaming** — token-by-token delivery, which is what makes the chat read as
   a chat rather than a form.
6. **streaming usage** — whether usage arrives in the stream. Providers only send
   it when asked (`stream_options`), and without it live cost reads as zero.

Nothing is inferred from the model's name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

TOOL = {
    "type": "function",
    "function": {
        "name": "get_scan_count",
        "description": "Return how many candidates the last scan produced.",
        "parameters": {
            "type": "object",
            "properties": {"sector": {"type": "string", "description": "sector name"}},
            "required": ["sector"],
        },
    },
}

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def auth(key: str) -> dict[str, str]:
    """Send no Authorization header at all when there is no key.

    An empty bearer is a malformed header, and httpx refuses it client-side —
    which hides the endpoint's real answer behind a local error.
    """
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def post(client, base_url: str, key: str, payload: dict, *, stream: bool = False):
    return client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=auth(key),
        json={**payload, "stream": stream},
        timeout=90.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default=config.LLM_BASE_URL)
    parser.add_argument("--api-key", default=config.LLM_API_KEY)
    parser.add_argument("--model", default=config.MODEL_TIERS["mid"] or "")
    args = parser.parse_args()

    if not args.api_key:
        print("\n  No API key. Set VC_LLM_API_KEY, put it in pipeline/.env, or pass --api-key.")
        print("  Probing anyway so the endpoint's own answer is visible.")

    if not (args.base_url and args.model):
        print("Need --base-url and --model (or VC_LLM_BASE_URL / VC_MODEL_MID).")
        raise SystemExit(2)

    import httpx

    findings: dict[str, object] = {}
    print(f"\n  endpoint : {args.base_url}\n  model    : {args.model}\n")

    with httpx.Client() as client:
        # 1 — reachable
        try:
            response = client.get(
                f"{args.base_url.rstrip('/')}/models",
                headers=auth(args.api_key),
                timeout=30.0,
            )
            ok = response.status_code == 200
            names = []
            if ok:
                names = [m.get("id") for m in (response.json().get("data") or [])]
            findings["reachable"] = ok
            print(f"  {'✓' if ok else '✗'} reachable            HTTP {response.status_code}")
            if names:
                shown = ", ".join(str(n) for n in names[:6])
                print(f"      models: {shown}{' …' if len(names) > 6 else ''}")
                if args.model not in names:
                    print(f"      ⚠ {args.model!r} is not in the list — check the name")
        except Exception as e:
            findings["reachable"] = False
            print(f"  ✗ reachable            {type(e).__name__}: {e}")

        base = {"model": args.model, "messages": [{"role": "user", "content": "Say OK."}]}

        # 2 — plain chat
        try:
            response = post(client, args.base_url, args.api_key, {**base, "max_tokens": 16})
            ok = response.status_code == 200
            findings["chat"] = ok
            text = ""
            if ok:
                text = (response.json()["choices"][0]["message"].get("content") or "")[:40]
            print(f"  {'✓' if ok else '✗'} chat                 HTTP {response.status_code}  {text!r}")
            if not ok:
                print(f"      {response.text[:200]}")
        except Exception as e:
            findings["chat"] = False
            print(f"  ✗ chat                 {type(e).__name__}: {e}")

        # 3 — tool calling
        try:
            response = post(client, args.base_url, args.api_key, {
                "model": args.model,
                "messages": [{"role": "user", "content": "How many candidates in the ai infra scan? Use the tool."}],
                "tools": [TOOL],
                "tool_choice": "auto",
                "max_tokens": 128,
            })
            calls = []
            if response.status_code == 200:
                calls = response.json()["choices"][0]["message"].get("tool_calls") or []
            ok = bool(calls)
            findings["function_calling"] = ok
            print(f"  {'✓' if ok else '✗'} tool calling         HTTP {response.status_code}  "
                  f"{calls[0]['function']['name'] if calls else 'no tool_calls returned'}")
            if response.status_code != 200:
                print(f"      {response.text[:200]}")
        except Exception as e:
            findings["function_calling"] = False
            print(f"  ✗ tool calling         {type(e).__name__}: {e}")

        # 4 — structured output
        try:
            response = post(client, args.base_url, args.api_key, {
                "model": args.model,
                "messages": [{"role": "user", "content": "Score this company 0-5 with a one-line reason."}],
                "response_format": SCHEMA,
                "max_tokens": 128,
            })
            ok = False
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"].get("content") or ""
                try:
                    parsed = json.loads(content)
                    ok = "score" in parsed and "reason" in parsed
                except json.JSONDecodeError:
                    ok = False
            findings["structured_output"] = ok
            print(f"  {'✓' if ok else '✗'} structured output    HTTP {response.status_code}")
            if response.status_code != 200:
                print(f"      {response.text[:200]}")
        except Exception as e:
            findings["structured_output"] = False
            print(f"  ✗ structured output    {type(e).__name__}: {e}")

        # 5/6 — streaming and streaming usage
        try:
            chunks, usage_seen = 0, False
            with client.stream(
                "POST", f"{args.base_url.rstrip('/')}/chat/completions",
                headers=auth(args.api_key),
                json={**base, "stream": True, "max_tokens": 32,
                      "stream_options": {"include_usage": True}},
                timeout=90.0,
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        frame = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    chunks += 1
                    if frame.get("usage"):
                        usage_seen = True
            findings["streaming"] = chunks > 0
            findings["stream_usage"] = usage_seen
            print(f"  {'✓' if chunks else '✗'} streaming            {chunks} chunks")
            print(f"  {'✓' if usage_seen else '✗'} streaming usage      "
                  f"{'usage in stream' if usage_seen else 'no usage — live cost will read as zero'}")
        except Exception as e:
            findings["streaming"] = False
            findings["stream_usage"] = False
            print(f"  ✗ streaming            {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ verdict
    print("\n  " + "─" * 60)
    if not findings.get("chat"):
        print("  The endpoint did not answer a plain completion. Nothing else matters until it does.")
        raise SystemExit(1)

    print("  Put this in your shell (or .env next to pipeline/):\n")
    print(f'    VC_LLM_BASE_URL="{args.base_url}"')
    print('    VC_LLM_API_KEY="…"')
    for tier in ("CHEAP", "MID", "STRONG"):
        print(f'    VC_MODEL_{tier}="{args.model}"')
    print(f'    VC_MODEL_FUNCTION_CALLING={int(bool(findings.get("function_calling")))}')
    print(f'    VC_MODEL_STRUCTURED_OUTPUT={int(bool(findings.get("structured_output")))}')
    print('    VC_MODEL_JSON_OUTPUT=1')
    print('    VC_MODEL_FAMILY="unknown"')

    print()
    if not findings.get("function_calling"):
        print("  ⚠ No tool calling. The live conversation cannot call tools and the")
        print("    enrichment analysts lose theirs. Declaring it anyway fails silently —")
        print("    the model will answer without ever calling anything.")
    if not findings.get("structured_output"):
        print("  ⚠ No structured output. The scorer and the memo writer use")
        print("    `output_content_type`; they will fail at the end of the funnel.")
    if findings.get("function_calling") and findings.get("structured_output"):
        print("  Both capabilities the agent layer needs are present.")
    print()


if __name__ == "__main__":
    main()
