# pi-anti-doom-loop

Stop agent doom loops in [pi](https://pi.dev/) before they burn tokens.

Cheap models sometimes get stuck repeating the same cheap tool call — `grep`
the same file, re-run the same failing command — with no progress. Each
iteration is so cheap nobody notices until the bill mounts. This extension
watches every tool call and blocks the loop at the source.

## Install

```bash
pi install npm:pi-anti-doom-loop
```

## What it detects

| Signal                                                  | Default                       | Blocked when                                                                                                                                         |
| ------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Same `(tool, args)` repeated                            | 3× in the last 10 calls       | The pattern has repeated `3` times with no change                                                                                                    |
| Same tool failing consecutively                         | 3×                            | A tool errored `3` times in a row — stop retrying it blindly                                                                                         |
| Same assistant text verbatim                            | 3× within the last N messages | The model re-emitted identical text `3` times inside the sliding window                                                                              |
| Same sentence inside ONE message                        | 3×                            | A sentence repeats `3`+ times within a single message (growing self-concatenation loops)                                                             |
| Near-identical text (rephrasing)                        | 3× in a row                   | Consecutive messages share ≥55% tokens — the model is rephrasing the same step                                                                       |
| Near-identical text cycle (rotating rephrased commands) | 3× within the last N messages | Near-identical assistant texts (≥55% token similarity) accumulate to the repeat threshold in the window, even when not identical and not consecutive |

Blocks hand the model an instructive reason ("change your approach, use a
different tool, or ask the user"). If the model ignores the block and re-issues
the exact same call, the turn is **aborted**.

### Escalation (text loops): steer → abort → bounded resume

The first text-loop detection **steers** the agent mid-run (injects guidance,
lets it continue). If it persists, the run is **aborted** and **one** fresh-
resume directive is queued so work continues with a new approach. If it still
loops after that, the run aborts for real and control returns to you — the
auto-resume budget is capped so a truly stuck model can't cycle forever.
Counters reset on every user prompt, so a task legitimately repeated later in
the same session is never a false positive.

## Configuration

Environment variables, read at session/prompt start:

| Variable                     | Default | Meaning                                                                                                                                            |
| ---------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PI_ANTI_LOOP_REPEATS`       | `3`     | Identical-call block threshold                                                                                                                     |
| `PI_ANTI_LOOP_FAILS`         | `3`     | Consecutive-failure block threshold                                                                                                                |
| `PI_ANTI_LOOP_TEXT_REPEATS`  | `3`     | Window/cycle repeat threshold for identical and near-identical assistant texts                                                                     |
| `PI_ANTI_LOOP_WINDOW`        | `10`    | How many recent calls/results are inspected                                                                                                        |
| `PI_ANTI_LOOP_TIME_WINDOW`   | `0`     | Elapsed-time window in ms (`0` = disabled, count-only): evicts window entries older than this so slow chronic loops over a long session are caught |
| `PI_ANTI_LOOP_FAIL_RATE`     | `0`     | Fail-rate block threshold `0..1` (`0` = disabled): block when a tool's error share of its in-window calls reaches this                             |
| `PI_ANTI_LOOP_FAIL_RATE_MIN` | `3`     | Minimum calls before the fail-rate window can block                                                                                                |
| `PI_ANTI_LOOP_TOOLS_EXCLUDE` | —       | Comma-separated tool names to disable detection for entirely (never block, never enter the window)                                                 |
| `PI_ANTI_LOOP_DISABLE`       | —       | Set to `1` to disable the extension entirely                                                                                                       |

## Command

- `/loopcheck` — show thresholds, counters (steers/aborts this session), suspend state, the current window contents (most-repeated recent calls and texts), wasted-token count, and the fail-rate/time-window/exclude config when enabled
- `/loopcheck reset` — clear counters
- `/loopcheck suspend` — pause detection until the next prompt (escape hatch for intentional repetition)
- `/loopcheck resume` — re-enable detection early

## Token-cost awareness

The detector estimates tokens burned on redundant repeats (~4 chars/token) and
reports `"~N tokens burned on repeats."` in tool-call block reasons. The
cumulative wasted-token count also appears in `/loopcheck` status, so you can
see how much a loop actually cost before it was stopped.

## How it works

Everything hooks into the `tool_call` / `tool_result` / `message_end` events;
detection is a small sliding-window counter (see `extensions/detector.ts`) with
per-session counters (steers/aborts) tracked in `extensions/controller.ts`.
Works with any model — cheap models just trigger it more often.

## Development

Requires Node 22.18+ (plain `node` runs the TS self-check).

```bash
npm install
npm test        # node --test: unit + fixture + fuzz + integration + e2e
npm run check   # npm test + tsc + oxlint --deny-warnings + oxfmt
```

### Test suite (Node built-in runner, no framework)

| Suite       | File                                          | What it proves                                                                                                    |
| ----------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| unit        | `tests/unit.test.ts`                          | detector semantics: repeat/failure/text signals, window eviction, options clamping, helpers                       |
| fixture     | `tests/fixtures.ts` + `tests/fixture.test.ts` | real doom-loop transcripts (CI-log loops, verbatim repeats) are caught; healthy sessions are not                  |
| fuzz        | `tests/fuzz.test.ts`                          | seeded random streams: never throws, no false positives, injected loops always block, canonical stability         |
| integration | `tests/integration.test.ts`                   | controller + `index.ts` adapter driven through a fake `PiLike`: blocks, escalations, aborts, resets, `/loopcheck` |
| e2e         | `tests/e2e.test.ts`                           | real subprocesses: detector self-check, version guard, tarball contents (extensions/scripts ship, tests don't)    |

> `peerDependencies` pins `@earendil-works/pi-coding-agent` at `"*"` on purpose — the
> [pi packages docs](https://pi.dev/docs/latest/packages) require an unbounded range for
> pi-core packages (pi provides them at runtime). The extension loads `.ts` directly via
> jiti, so no build step ships; `prepublishOnly` runs the full quality gate before publish.

> When a call is blocked, escalation still works without recording it in the window:
> re-issuing the identical call increments a per-signature block counter and aborts the
> turn on the second block. Thresholds are clamped to a minimum of 2 so a bad config
> can never brick the agent.

## Releasing

Publishing is handled by the GitHub Actions workflow [`.github/workflows/release.yml`](.github/workflows/release.yml), guarded against version drift:

1. Add an npm **Automation** token as the `NPM_TOKEN` repo secret (Settings → Secrets and variables → Actions, or `gh secret set NPM_TOKEN`).
2. Bump `version` in `package.json`, commit, then tag and push:

```bash
git tag v0.0.1
git push origin v0.0.1
```

CI runs the quality gate, then the **version bump guard** (`npm run guard`): it blocks publishing if the version is already on npm or the tag doesn't match `package.json`. After the first publish, the [pi.dev gallery](https://pi.dev/packages) picks the package up automatically via the `pi-package` keyword.

## License

MIT