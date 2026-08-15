"""Scheduled work: fresh sessions, a legible threshold, and gated delivery."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import live
from gateway import cron
from gateway import sessions as sessions_module


def report(company: str, *, changes=(), failed=()) -> live.LiveReport:
    """A real `LiveReport`, not a stand-in — the threshold reads its actual shape."""
    result = live.LiveReport(
        company=company,
        checked_at=datetime.now(timezone.utc),
        scan_seen_until=None,
        changes=list(changes),
    )
    for source in failed:
        result.checks.append(live.SourceCheck(source, False, error="429 rate limited"))
    return result


def scheduler(**kwargs) -> cron.Scheduler:
    store = cron.CronStore(Path(tempfile.mkdtemp(prefix="cron-")) / "cron.json")
    sessions = sessions_module.SessionManager(
        store=sessions_module.SessionStore(Path(tempfile.mkdtemp(prefix="cron-sessions-")))
    )
    return cron.Scheduler(store=store, sessions=sessions, **kwargs)


class ThresholdTests(unittest.TestCase):
    def test_a_big_star_move_is_notable(self) -> None:
        findings = cron.Threshold(star_delta=25).judge(
            report("Acme", changes=["acme/db: stars 1,200 → 1,340 (+140)"])
        )
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].notable)

    def test_small_drift_is_recorded_but_not_notable(self) -> None:
        """Repositories gain a few stars a day. That is not news."""
        findings = cron.Threshold(star_delta=25).judge(
            report("Acme", changes=["acme/db: stars 1,200 → 1,203 (+3)"])
        )
        self.assertEqual(len(findings), 1)
        self.assertFalse(findings[0].notable)

    def test_a_star_loss_counts_too(self) -> None:
        findings = cron.Threshold(star_delta=25).judge(
            report("Acme", changes=["acme/db: stars 1,340 → 1,200 (-140)"])
        )
        self.assertTrue(findings[0].notable)

    def test_a_non_star_change_is_notable(self) -> None:
        findings = cron.Threshold().judge(
            report("Acme", changes=["acme/db: pushed 2026-08-13, after the scan"])
        )
        self.assertTrue(findings[0].notable)

    def test_a_source_that_could_not_be_checked_is_reported_not_notified(self) -> None:
        """The distinction this project keeps insisting on, enforced here."""
        findings = cron.Threshold().judge(report("Acme", failed=["github"]))
        self.assertEqual(len(findings), 1)
        self.assertIn("could not be checked", findings[0].headline)
        self.assertFalse(findings[0].notable, "an outage is not news about a company")
        self.assertIn("429", findings[0].detail)

    def test_failures_and_changes_are_both_kept(self) -> None:
        findings = cron.Threshold().judge(
            report("Acme", changes=["acme/db: pushed 2026-08-13, after the scan"], failed=["hn"])
        )
        self.assertEqual(len(findings), 2)
        self.assertEqual(sum(1 for f in findings if f.notable), 1)

    def test_ignored_terms_are_dropped(self) -> None:
        threshold = cron.Threshold(ignore=("dependabot",))
        findings = threshold.judge(report("Acme", changes=["acme/db: dependabot pushed"]))
        self.assertEqual(findings, [])


class ScheduleTests(unittest.TestCase):
    def test_a_new_job_is_due_immediately(self) -> None:
        job = cron.Job(id="w", kind="live", every_minutes=60)
        self.assertTrue(job.due())

    def test_a_job_is_not_due_again_until_its_interval_passes(self) -> None:
        job = cron.Job(id="w", kind="live", every_minutes=60)
        job.last_run_at = datetime.now(timezone.utc).isoformat()
        self.assertFalse(job.due())
        self.assertTrue(job.due(datetime.now(timezone.utc) + timedelta(minutes=61)))

    def test_a_disabled_job_is_never_due(self) -> None:
        self.assertFalse(cron.Job(id="w", kind="live", every_minutes=1, enabled=False).due())

    def test_jobs_survive_a_reload(self) -> None:
        first = scheduler()
        first.watch(["Acme", "Argonix"], every_minutes=120)

        again = cron.Scheduler(store=cron.CronStore(first.store.path))
        self.assertIn("watchlist", again.jobs)
        self.assertEqual(again.jobs["watchlist"].companies, ["Acme", "Argonix"])

    def test_a_corrupt_job_file_does_not_stop_the_gateway(self) -> None:
        store = cron.CronStore(Path(tempfile.mkdtemp(prefix="cron-")) / "cron.json")
        store.path.write_text("{ truncated", encoding="utf-8")
        self.assertEqual(cron.Scheduler(store=store).jobs, {})


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_run_gets_its_own_session(self) -> None:
        """A scheduled job that reuses context grows more expensive and more wrong."""
        sched = scheduler()
        job = sched.add(cron.Job(id="w", kind="live", every_minutes=1, companies=[]))

        first = await sched.run(job)
        second = await sched.run(job)
        self.assertNotEqual(first["session"], second["session"])
        self.assertIn(":cron:", first["session"])

    async def test_a_failing_job_is_recorded_rather_than_raised(self) -> None:
        sched = scheduler()
        job = sched.add(cron.Job(id="bad", kind="nonsense", every_minutes=1))
        outcome = await sched.run(job)
        self.assertTrue(outcome["status"].startswith("error"))
        self.assertEqual(sched.jobs["bad"].last_status, outcome["status"])

    async def test_notification_only_fires_for_notable_findings(self) -> None:
        sent: list[str] = []
        sched = scheduler(notifier=lambda text: sent.append(text) or True)
        sched.threshold = cron.Threshold(star_delta=25)
        job = sched.add(cron.Job(id="w", kind="live", every_minutes=1))

        sched._work = lambda j: _findings([cron.Finding("Acme", "small", notable=False)])
        await sched.run(job)
        self.assertEqual(sent, [], "a non-notable finding must not interrupt anyone")

        sched._work = lambda j: _findings([cron.Finding("Acme", "+140 stars", notable=True)])
        await sched.run(job)
        self.assertEqual(len(sent), 1)
        self.assertIn("Acme", sent[0])

    async def test_a_job_with_notify_off_stays_quiet(self) -> None:
        sent: list[str] = []
        sched = scheduler(notifier=lambda text: sent.append(text) or True)
        job = sched.add(cron.Job(id="w", kind="live", every_minutes=1, notify=False))
        sched._work = lambda j: _findings([cron.Finding("Acme", "big", notable=True)])

        await sched.run(job)
        self.assertEqual(sent, [])

    async def test_no_notifier_means_no_delivery_and_no_crash(self) -> None:
        sched = scheduler(notifier=None)
        job = sched.add(cron.Job(id="w", kind="live", every_minutes=1))
        sched._work = lambda j: _findings([cron.Finding("Acme", "big", notable=True)])

        outcome = await sched.run(job)
        self.assertFalse(outcome["delivered"])
        self.assertEqual(outcome["status"], "ok")

    async def test_a_failing_notifier_does_not_fail_the_run(self) -> None:
        def broken(text: str):
            raise RuntimeError("telegram down")

        sched = scheduler(notifier=broken)
        job = sched.add(cron.Job(id="w", kind="live", every_minutes=1))
        sched._work = lambda j: _findings([cron.Finding("Acme", "big", notable=True)])

        outcome = await sched.run(job)
        self.assertEqual(outcome["status"], "ok")
        self.assertFalse(outcome["delivered"])

    async def test_tick_runs_only_what_is_due(self) -> None:
        sched = scheduler()
        sched.add(cron.Job(id="due", kind="live", every_minutes=1))
        later = sched.add(cron.Job(id="not-due", kind="live", every_minutes=60))
        later.last_run_at = datetime.now(timezone.utc).isoformat()

        results = await sched.tick()
        self.assertEqual([r["job"] for r in results], ["due"])


async def _findings(items):
    return items


if __name__ == "__main__":
    unittest.main()
