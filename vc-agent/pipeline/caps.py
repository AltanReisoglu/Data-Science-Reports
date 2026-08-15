"""Capabilities: typed contracts with a fallback, and no plugin loader.

OpenClaw's extension model rests on one distinction (docs/13 §8.2):

> **plugin = ownership boundary · capability = core contract**

and one instruction that follows from it: when adding a new area, the first
question is not *"which provider should we hardcode?"* but *"what is the core
capability contract?"*

That distinction is worth taking. The machinery around it mostly is not, and
saying why is more useful than quietly building half of it.

**What is taken.** Named contracts (`typing.Protocol`), a registry that can hold
more than one implementation of each, a **declared fallback**, and
**quarantine** — a provider that raises is disabled and the fallback takes over,
because OpenClaw's own rule for a broken context engine is that *the agent does
not go silent*.

**What is not, and why.** No manifest format, no discovery pass, no activation
pipeline, no in-process plugin loading. OpenClaw needs those because third
parties ship plugins into it. Nobody ships plugins into this. Building a loader
for a set of implementations that all live in this repository would be ceremony
that has to be maintained and cannot be exercised. If that changes, the seam is
one function — `Registry.discover` — and the rest of this file does not move.

There is also a warning in OpenClaw's own documentation worth carrying over:
native plugins run **in-process, unsandboxed**, at the same trust level as the
core. Anything registered here has the same property. That is fine while every
implementation is ours and is stated here so it stays a decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

log = logging.getLogger("vcagent.caps")


# --------------------------------------------------------------------------- contracts


@runtime_checkable
class Collector(Protocol):
    """A source of signals. `collectors/base.Collector` already satisfies this."""

    name: str

    def run(self, *, query: str, days: int) -> Any: ...


@runtime_checkable
class ContextEngine(Protocol):
    """Decides what the model sees. `context_engine` is the built-in."""

    async def add_message(self, message: Any) -> None: ...

    async def get_messages(self) -> list[Any]: ...


@runtime_checkable
class MemoryBackend(Protocol):
    """Where remembered things live. `memory` is Markdown files in a workspace."""

    def search(self, query: str, k: int = 5) -> list[Any]: ...

    def note(self, text: str, *, tag: str = "", day: str | None = None) -> Any: ...

    def preamble(self, limit: int = 4000) -> str: ...


@runtime_checkable
class Notifier(Protocol):
    """Somewhere to deliver an alert. The OpenClaw bridge is the only one today."""

    def __call__(self, message: str) -> Any: ...


@runtime_checkable
class ToolProvider(Protocol):
    """Something that contributes callables to an agent."""

    def build(self, sources: Any) -> list[Callable[..., Any]]: ...


@runtime_checkable
class Channel(Protocol):
    """A surface a turn can arrive from and be delivered to.

    Two halves, and the second is the one that did not exist before `channels.py`:
    routing already knew a turn's origin, but nothing could *send* anywhere except
    back down the request that asked. `may_send_to` is access control, and it
    belongs to the channel rather than to the agent that asked it to send.
    """

    name: str

    def may_send_to(self, peer: str) -> bool: ...

    async def send(self, message: Any) -> Any: ...


CONTRACTS: dict[str, type] = {
    "channel": Channel,
    "collector": Collector,
    "context_engine": ContextEngine,
    "memory": MemoryBackend,
    "notifier": Notifier,
    "tools": ToolProvider,
}


# --------------------------------------------------------------------------- registry


@dataclass
class Entry:
    capability: str
    name: str
    factory: Callable[[], Any]
    fallback: bool = False
    failures: int = 0
    quarantined: bool = False
    last_error: str = ""


class Registry:
    """Implementations per capability, with a fallback that cannot be quarantined."""

    def __init__(self, *, failure_limit: int = 2) -> None:
        self._entries: dict[str, dict[str, Entry]] = {}
        self._selected: dict[str, str] = {}
        self.failure_limit = failure_limit

    # ------------------------------------------------------------ registration

    def register(
        self,
        capability: str,
        name: str,
        factory: Callable[[], Any],
        *,
        fallback: bool = False,
        select: bool = False,
    ) -> Entry:
        if capability not in CONTRACTS:
            raise ValueError(
                f"unknown capability {capability!r}; known: {', '.join(sorted(CONTRACTS))}"
            )
        entry = Entry(capability=capability, name=name, factory=factory, fallback=fallback)
        self._entries.setdefault(capability, {})[name] = entry
        if select or (fallback and capability not in self._selected):
            self._selected[capability] = name
        return entry

    def select(self, capability: str, name: str) -> None:
        if name not in self._entries.get(capability, {}):
            raise KeyError(f"{name!r} is not registered for {capability!r}")
        self._selected[capability] = name

    def names(self, capability: str) -> list[str]:
        return sorted(self._entries.get(capability, {}))

    def fallback_for(self, capability: str) -> Entry | None:
        return next(
            (e for e in self._entries.get(capability, {}).values() if e.fallback), None
        )

    # ------------------------------------------------------------ resolution

    def get(self, capability: str) -> Any:
        """Build the selected implementation, dropping to the fallback if it fails.

        The fallback is exempt from quarantine on purpose: if it is also broken
        there is nothing left to fall back to, and pretending otherwise would
        hide the real failure behind a `None`.
        """
        entries = self._entries.get(capability, {})
        if not entries:
            raise LookupError(f"no implementation registered for {capability!r}")

        chosen = self._selected.get(capability)
        order = [chosen] if chosen else []
        fallback = self.fallback_for(capability)
        if fallback is not None and fallback.name != chosen:
            order.append(fallback.name)

        last: Exception | None = None
        for name in order:
            entry = entries.get(name)
            if entry is None or (entry.quarantined and not entry.fallback):
                continue
            try:
                return entry.factory()
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._quarantine(entry, exc)

        raise LookupError(f"no usable implementation for {capability!r}") from last

    def _quarantine(self, entry: Entry, exc: Exception) -> None:
        entry.failures += 1
        entry.last_error = f"{type(exc).__name__}: {exc}"
        log.warning("capability %s/%s failed: %s", entry.capability, entry.name, entry.last_error)
        if entry.fallback:
            return
        if entry.failures >= self.failure_limit:
            entry.quarantined = True
            log.warning(
                "capability %s/%s quarantined; falling back", entry.capability, entry.name
            )

    def revive(self, capability: str, name: str) -> bool:
        entry = self._entries.get(capability, {}).get(name)
        if entry is None or not entry.quarantined:
            return False
        entry.quarantined = False
        entry.failures = 0
        return True

    # ------------------------------------------------------------ inspection

    def report(self) -> dict[str, Any]:
        return {
            capability: {
                "selected": self._selected.get(capability),
                "implementations": [
                    {
                        "name": e.name,
                        "fallback": e.fallback,
                        "quarantined": e.quarantined,
                        "failures": e.failures,
                        "last_error": e.last_error,
                    }
                    for e in sorted(entries.values(), key=lambda x: x.name)
                ],
            }
            for capability, entries in sorted(self._entries.items())
        }

    def discover(self) -> int:
        """The seam a plugin loader would occupy. Deliberately empty — see the module docstring."""
        return 0

    def __iter__(self) -> Iterator[Entry]:
        return iter(e for entries in self._entries.values() for e in entries.values())


REGISTRY = Registry()


def install_defaults(registry: Registry | None = None) -> Registry:
    """Register what this repository actually provides. Called by the gateway."""
    target = registry or REGISTRY

    def legacy_context():
        import context_engine

        return context_engine.legacy_context()

    def compacting_context():
        import context_engine

        return context_engine.CompactingChatCompletionContext()

    def markdown_memory():
        import memory

        return memory

    def shared_tools():
        from gateway import tools as tools_module

        return tools_module

    # `legacy` is the fallback for the same reason OpenClaw keeps one: if the
    # richer engine cannot be built, a plain buffered context still answers.
    target.register("context_engine", "legacy", legacy_context, fallback=True)
    target.register("context_engine", "compacting", compacting_context, select=True)
    target.register("memory", "markdown", markdown_memory, fallback=True)
    target.register("tools", "gateway", shared_tools, fallback=True)
    return target


__all__ = [
    "CONTRACTS", "REGISTRY", "Collector", "ContextEngine", "Entry", "MemoryBackend",
    "Notifier", "Registry", "ToolProvider", "install_defaults",
]
