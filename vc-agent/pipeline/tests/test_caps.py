"""Capability contracts: fallback, quarantine, and the fallback's exemption."""

from __future__ import annotations

import unittest

import caps


class ContractTests(unittest.TestCase):
    def test_the_real_collectors_satisfy_the_collector_contract(self) -> None:
        """A contract nothing implements is a guess. These are the shipped ones."""
        from collectors.github import GitHub
        from collectors.hackernews import HackerNews

        for cls in (GitHub, HackerNews):
            with self.subTest(collector=cls.__name__):
                self.assertIsInstance(cls(), caps.Collector)

    def test_the_context_engine_satisfies_its_contract(self) -> None:
        import context_engine

        self.assertIsInstance(
            context_engine.CompactingChatCompletionContext(), caps.ContextEngine
        )

    def test_the_memory_module_satisfies_its_contract(self) -> None:
        import memory

        self.assertIsInstance(memory, caps.MemoryBackend)

    def test_registering_an_unknown_capability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            caps.Registry().register("telepathy", "x", lambda: None)


class ResolutionTests(unittest.TestCase):
    def test_the_selected_implementation_is_used(self) -> None:
        registry = caps.Registry()
        registry.register("memory", "plain", lambda: "plain", fallback=True)
        registry.register("memory", "fancy", lambda: "fancy", select=True)
        self.assertEqual(registry.get("memory"), "fancy")

    def test_a_fallback_is_selected_when_nothing_else_is(self) -> None:
        registry = caps.Registry()
        registry.register("memory", "plain", lambda: "plain", fallback=True)
        self.assertEqual(registry.get("memory"), "plain")

    def test_a_broken_implementation_falls_back_rather_than_raising(self) -> None:
        """OpenClaw's rule for a crashed context engine: the agent does not go silent."""
        registry = caps.Registry()
        registry.register("context_engine", "legacy", lambda: "legacy", fallback=True)
        registry.register(
            "context_engine", "broken", _raises(RuntimeError("boom")), select=True
        )
        self.assertEqual(registry.get("context_engine"), "legacy")

    def test_repeated_failures_quarantine_it(self) -> None:
        registry = caps.Registry(failure_limit=2)
        registry.register("context_engine", "legacy", lambda: "legacy", fallback=True)
        broken = registry.register(
            "context_engine", "broken", _raises(ValueError("bad")), select=True
        )

        for _ in range(3):
            registry.get("context_engine")

        self.assertTrue(broken.quarantined)
        self.assertIn("bad", broken.last_error)

    def test_the_fallback_is_never_quarantined(self) -> None:
        """If the last resort could be disabled, a failure would surface as `None`."""
        registry = caps.Registry(failure_limit=1)
        fallback = registry.register(
            "memory", "plain", _raises(RuntimeError("also broken")), fallback=True
        )
        for _ in range(3):
            with self.assertRaises(LookupError):
                registry.get("memory")

        self.assertFalse(fallback.quarantined)
        self.assertGreaterEqual(fallback.failures, 3)

    def test_a_missing_capability_says_so(self) -> None:
        with self.assertRaises(LookupError):
            caps.Registry().get("memory")

    def test_revive_brings_a_fixed_implementation_back(self) -> None:
        registry = caps.Registry(failure_limit=1)
        registry.register("memory", "plain", lambda: "plain", fallback=True)
        registry.register("memory", "flaky", _raises(RuntimeError("x")), select=True)
        registry.get("memory")

        self.assertTrue(registry.revive("memory", "flaky"))
        self.assertFalse(registry.revive("memory", "flaky"), "already healthy")


class DefaultsTests(unittest.TestCase):
    def test_the_defaults_build(self) -> None:
        registry = caps.install_defaults(caps.Registry())
        self.assertIsNotNone(registry.get("context_engine"))
        self.assertIsNotNone(registry.get("memory"))
        self.assertIsNotNone(registry.get("tools"))

    def test_the_context_engine_has_a_declared_fallback(self) -> None:
        registry = caps.install_defaults(caps.Registry())
        fallback = registry.fallback_for("context_engine")
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.name, "legacy")

    def test_the_report_names_what_is_selected(self) -> None:
        report = caps.install_defaults(caps.Registry()).report()
        self.assertEqual(report["context_engine"]["selected"], "compacting")
        self.assertIn(
            "legacy", [i["name"] for i in report["context_engine"]["implementations"]]
        )

    def test_discovery_is_deliberately_empty(self) -> None:
        """No plugin loader — the module docstring explains why, this pins it."""
        self.assertEqual(caps.Registry().discover(), 0)


def _raises(exc: Exception):
    def factory():
        raise exc

    return factory


if __name__ == "__main__":
    unittest.main()
