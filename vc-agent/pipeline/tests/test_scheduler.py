"""Scheduling, which this project delegates rather than implements.

The translation is the whole module, so the translation is what these check: our
three `when` forms in, OpenClaw's schema out, and the two payload kinds that are
deliberately never produced.

Nothing here talks to a Gateway. `build_job` and `parse_when` are pure, and that
is the reason they were split out of `create` — a scheduler you can only test by
scheduling something is a scheduler nobody tests.
"""

from __future__ import annotations

import unittest

import scheduler


class WhenTest(unittest.TestCase):
    def test_daily_becomes_a_cron_expression_with_a_timezone(self) -> None:
        self.assertEqual(
            scheduler.parse_when("her gün 09:00"),
            {"kind": "cron", "expr": "0 9 * * *", "tz": scheduler.TZ},
        )
        # Minutes are not dropped, and the field order is cron's, not ours.
        self.assertEqual(scheduler.parse_when("her gün 03:45")["expr"], "45 3 * * *")

    def test_interval_becomes_milliseconds(self) -> None:
        self.assertEqual(scheduler.parse_when("30dk"), {"kind": "every", "everyMs": 1_800_000})
        self.assertEqual(scheduler.parse_when("2saat"), {"kind": "every", "everyMs": 7_200_000})
        self.assertEqual(scheduler.parse_when("3 gün"), {"kind": "every", "everyMs": 259_200_000})

    def test_after_becomes_a_one_shot_timestamp(self) -> None:
        got = scheduler.parse_when("20dk sonra")
        self.assertEqual(got["kind"], "at")
        self.assertTrue(got["at"].endswith("Z"), got["at"])

    def test_a_form_we_do_not_accept_names_the_ones_we_do(self) -> None:
        """A scheduler that guesses fires at a time nobody chose."""
        with self.assertRaises(scheduler.WhenError) as caught:
            scheduler.parse_when("yarın öğlen")
        message = str(caught.exception)
        for form in ("her gün 09:00", "30dk", "20dk sonra"):
            self.assertIn(form, message)

    def test_a_zero_interval_is_refused_and_one_minute_is_the_floor(self) -> None:
        """`0dk` parses cleanly and would mean "as fast as possible"."""
        with self.assertRaises(scheduler.WhenError):
            scheduler.parse_when("0dk")
        self.assertEqual(scheduler.parse_when("1dk")["everyMs"], 60_000)

    def test_out_of_range_clock_time_is_refused(self) -> None:
        with self.assertRaises(scheduler.WhenError):
            scheduler.parse_when("her gün 25:00")


class BuildJobTest(unittest.TestCase):
    JOB = ("gece taraması", "her gün 03:00", "vc_start_scan tool'unu çağır")

    def test_the_payload_is_always_an_agent_turn(self) -> None:
        """`command` and `script` payloads are shell, and shell is the gate's call.

        A job definition that runs unattended at 3am is the worst place to decide
        about shell access, so this module cannot express it.
        """
        payload = scheduler.build_job(*self.JOB)["payload"]
        self.assertEqual(payload["kind"], "agentTurn")
        self.assertNotIn("argv", payload)
        self.assertNotIn("script", payload)

    def test_runs_are_isolated_by_default(self) -> None:
        job = scheduler.build_job(*self.JOB)
        self.assertEqual(job["sessionTarget"], "isolated")
        self.assertEqual(job["wakeMode"], "now")

    def test_the_schedule_is_the_parsed_one(self) -> None:
        job = scheduler.build_job(*self.JOB)
        self.assertEqual(job["schedule"], scheduler.parse_when(self.JOB[1]))

    def test_a_job_without_a_name_or_a_task_is_refused(self) -> None:
        with self.assertRaises(scheduler.WhenError):
            scheduler.build_job("  ", "30dk", "bir şey")
        with self.assertRaises(scheduler.WhenError):
            scheduler.build_job("ad", "30dk", "")

    def test_event_kinds_are_out_of_scope(self) -> None:
        """`on-exit` and `stream` supervise a command; not from here."""
        self.assertNotIn("on-exit", scheduler.KINDS)
        self.assertNotIn("stream", scheduler.KINDS)


class CommandTest(unittest.TestCase):
    """`/openclaw schedule …` — the line, not the API call.

    This subcommand exists because prose did not work: asked in a sentence,
    OpenClaw's agent replied with a clarifying question instead of a job. These
    check that the typed form either does the thing or names the syntax — it
    never asks.
    """

    def test_an_empty_line_lists(self) -> None:
        self.assertEqual(scheduler.parse_command(""), {"action": "list"})
        self.assertEqual(scheduler.parse_command("   "), {"action": "list"})

    def test_create_splits_on_the_bar(self) -> None:
        got = scheduler.parse_command("her gün 05:00 | bana merhaba de")
        self.assertEqual(got["action"], "create")
        self.assertEqual(got["when"], "her gün 05:00")
        self.assertEqual(got["ask"], "bana merhaba de")
        self.assertEqual(got["to"], "")

    def test_a_delivery_target_is_optional_and_parsed(self) -> None:
        got = scheduler.parse_command("30dk | durumu söyle > telegram:123")
        self.assertEqual(got["to"], "telegram:123")
        self.assertEqual(got["ask"], "durumu söyle")

    def test_remove_takes_an_id(self) -> None:
        self.assertEqual(scheduler.parse_command("sil abc123"),
                         {"action": "remove", "id": "abc123"})
        with self.assertRaises(scheduler.WhenError):
            scheduler.parse_command("sil")

    def test_a_missing_bar_names_the_syntax(self) -> None:
        with self.assertRaises(scheduler.WhenError) as caught:
            scheduler.parse_command("her gün 05:00 bana merhaba de")
        self.assertIn("|", str(caught.exception))

    def test_an_empty_task_is_refused(self) -> None:
        with self.assertRaises(scheduler.WhenError):
            scheduler.parse_command("her gün 05:00 |")


class DeliveryTest(unittest.TestCase):
    def test_no_target_means_no_delivery_block(self) -> None:
        """Guessing an address to send to is the kind of helpful that mails a stranger."""
        self.assertNotIn("delivery", scheduler.build_job("x", "30dk", "y"))

    def test_a_target_becomes_an_announce_block(self) -> None:
        job = scheduler.build_job("x", "30dk", "y", to="telegram:456")
        self.assertEqual(job["delivery"],
                         {"mode": "announce", "channel": "telegram", "to": "456"})

    def test_a_target_without_an_address_is_refused(self) -> None:
        with self.assertRaises(scheduler.WhenError):
            scheduler.build_job("x", "30dk", "y", to="telegram")


class DescribeTest(unittest.TestCase):
    def test_a_schedule_survives_the_round_trip(self) -> None:
        for when in ("her gün 09:00", "30dk", "2saat", "3 gün"):
            with self.subTest(when=when):
                self.assertEqual(
                    scheduler.describe(scheduler.parse_when(when)),
                    scheduler.describe(scheduler.parse_when(when)),
                )
        self.assertEqual(scheduler.describe(scheduler.parse_when("her gün 09:00")),
                         "her gün 09:00")
        self.assertEqual(scheduler.describe(scheduler.parse_when("90dk")),
                         "90 dakikada bir")

    def test_an_unknown_shape_does_not_pretend(self) -> None:
        self.assertEqual(scheduler.describe({"kind": "on-exit"}), "on-exit")
        self.assertEqual(scheduler.describe({}), "?")


class LimitTest(unittest.TestCase):
    def test_the_linger_caveat_is_carried_not_remembered(self) -> None:
        """The one failure a scheduler must not have is stopping quietly.

        The Gateway here is a systemd *user* service with `Linger=no`, so it dies
        with the session. That is not a footnote for a doc nobody rereads; it
        rides along with every listing.
        """
        self.assertIn("Linger", scheduler.LINGER_NOTE)


if __name__ == "__main__":
    unittest.main()
