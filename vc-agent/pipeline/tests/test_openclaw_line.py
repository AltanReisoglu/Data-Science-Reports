"""The `/openclaw` escape hatch: parsing a typed line, and what it refuses.

This path has no model in it, so there is nothing to blame a bad call on but the
parser. Every case here is something a person can actually type into the chat box
at two in the morning, including the ones that are wrong.
"""

from __future__ import annotations

import unittest

import openclaw_control as oc


class ParseLineTest(unittest.TestCase):
    def test_method_alone(self) -> None:
        method, params, error = oc.parse_line("sessions.list")
        self.assertEqual((method, params, error), ("sessions.list", {}, ""))

    def test_the_prefix_is_optional(self) -> None:
        """The web sends the whole line; a caller with the prefix stripped is fine too."""
        with_prefix = oc.parse_line("/openclaw cron.list")
        without = oc.parse_line("cron.list")
        self.assertEqual(with_prefix, without)

    def test_json_params_are_parsed(self) -> None:
        method, params, error = oc.parse_line('/openclaw cron.run {"id": "watch"}')
        self.assertEqual(method, "cron.run")
        self.assertEqual(params, {"id": "watch"})
        self.assertEqual(error, "")

    def test_params_with_spaces_survive(self) -> None:
        """`partition` splits once, so JSON keeps its spaces instead of being chopped."""
        _method, params, error = oc.parse_line('chat.send {"text": "merhaba dunya"}')
        self.assertEqual(error, "")
        self.assertEqual(params["text"], "merhaba dunya")

    def test_broken_json_explains_itself(self) -> None:
        _method, _params, error = oc.parse_line('cron.run {"id": }')
        self.assertIn("not JSON", error)
        self.assertIn("position", error, "the operator needs to know where to look")

    def test_a_json_list_is_refused(self) -> None:
        """Valid JSON, wrong shape. `--params` wants an object."""
        _method, _params, error = oc.parse_line('cron.run ["watch"]')
        self.assertIn("must be a JSON object", error)
        self.assertIn("list", error)

    def test_an_empty_line_answers_with_usage(self) -> None:
        for line in ("", "   ", "/openclaw", "/openclaw   "):
            _method, _params, error = oc.parse_line(line)
            self.assertIn("Usage", error, f"no usage for {line!r}")


class RunLineTest(unittest.IsolatedAsyncioTestCase):
    async def test_methods_answers_without_the_binary(self) -> None:
        """The one question that must work even when OpenClaw is not installed.

        "What can I type here" is the first thing anybody asks, and answering it
        by shelling out would make the help unavailable exactly when the bridge
        is broken — which is when help is wanted.
        """
        outcome = await oc.run_line("/openclaw methods")
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["tier"], "local")
        self.assertIn("sessions.list", outcome["result"]["read"])
        # `cron.add` is the method the running Gateway actually has;
        # `cron.create` answers "unknown method".
        self.assertIn("cron.add", outcome["result"]["write"])

    async def test_a_forbidden_method_never_reaches_the_binary(self) -> None:
        outcome = await oc.run_line("/openclaw config.set {}")
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["tier"], "forbidden")
        self.assertIn("no per-call approval", outcome["error"])

    async def test_a_typo_does_not_shell_out(self) -> None:
        """A parse failure has to come back before any subprocess is started."""
        outcome = await oc.run_line('cron.run {"id": }')
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["method"], "cron.run")
        self.assertIn("not JSON", outcome["error"])


class RoutingTest(unittest.TestCase):
    """Method or sentence? The dot decides, and the cost of each mistake differs."""

    def test_gateway_methods_are_recognised(self) -> None:
        for token in ("sessions.list", "chat.send", "config.set", "node.list"):
            self.assertTrue(oc.looks_like_a_method(token), token)

    def test_sentences_are_not(self) -> None:
        for token in ("adın", "merhaba", "kaç", "what"):
            self.assertFalse(oc.looks_like_a_method(token), token)

    def test_a_misspelt_method_stays_a_method(self) -> None:
        """The asymmetry that picked this rule.

        Sending `sessions.lst` to the agent as a sentence would swallow a typo and
        answer with a friendly paraphrase. Keeping it a method produces "unknown
        method: sessions.lst", which is the sentence the operator needs to read.
        """
        self.assertTrue(oc.looks_like_a_method("sessions.lst"))
        self.assertTrue(oc.looks_like_a_method("cron.crate"))


class ChatReplyTest(unittest.TestCase):
    def test_text_parts_are_joined_and_others_dropped(self) -> None:
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Henüz bir adım yok"},
                {"type": "thinking", "text": "gizli"},
                {"type": "text", "text": "sen ne dersin?"},
            ],
        }
        self.assertEqual(oc._reply_text(message), "Henüz bir adım yok\nsen ne dersin?")

    def test_a_still_running_row_reads_as_empty(self) -> None:
        """OpenClaw writes an empty assistant row while the run is going.

        Polling stops at the first assistant row, so if this returned anything
        truthy the screen would print a blank bubble and call the turn finished.
        """
        self.assertEqual(oc._reply_text({"role": "assistant", "content": []}), "")
        self.assertEqual(oc._reply_text({"role": "assistant", "content": None}), "")
        self.assertEqual(
            oc._reply_text({"role": "assistant", "content": [{"type": "text", "text": "  "}]}), ""
        )

    def test_plain_string_content_still_works(self) -> None:
        self.assertEqual(oc._reply_text({"content": " merhaba "}), "merhaba")

    def test_the_chat_session_is_ours_alone(self) -> None:
        """It must not collide with OpenClaw's dashboard or channel sessions.

        Those look like `agent:main:dashboard:<uuid>` and the channel ones carry a
        real peer. A question typed here landing in one of those would put our text
        into somebody's actual conversation.
        """
        self.assertTrue(oc.CHAT_SESSION_PREFIX.startswith("agent:main:"))
        self.assertNotIn("dashboard", oc.CHAT_SESSION_PREFIX)
        self.assertTrue(oc.CHAT_SESSION_PREFIX.endswith(":"))


if __name__ == "__main__":
    unittest.main()
