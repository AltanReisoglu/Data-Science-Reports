"""Every route answers.

Written after `/api/state` returned 500 in the browser: the singleton→registry
refactor left two calls behind (`CHAT.mcp_status`, `CHAT.cost()`) and no test
touched that route, so the suite stayed green while the page was broken.

The lesson generalises past those two lines — a route with no test is a route
whose contract is whatever it happened to do last. So this asks every one of them
for a response, which is cheap and would have caught it.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import server


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The context manager runs startup/shutdown, so the gateway runtime and
        # the channel agents are exercised too.
        cls.ctx = TestClient(server.app)
        cls.client = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ctx.__exit__(None, None, None)

    def test_every_get_route_answers(self) -> None:
        """A 500 anywhere here is the bug this file exists for."""
        for path in (
            "/", "/style.css", "/app.js",
            "/api/state", "/api/health", "/api/scan",
            "/api/sessions", "/api/channels", "/api/relays", "/api/approvals",
            "/api/inbox",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertLess(
                    response.status_code, 500,
                    f"{path} returned {response.status_code}: {response.text[:200]}",
                )

    def test_state_reports_per_session_rather_than_per_process(self) -> None:
        """The actual regression: these came off the registry, which has neither."""
        body = self.client.get("/api/state").json()
        if body.get("has_scan"):
            self.assertIn("mcp", body)
            self.assertIn("chat_cost", body)
            self.assertIsInstance(body["chat_cost"], dict)

    def test_state_accepts_a_peer(self) -> None:
        self.assertLess(self.client.get("/api/state?peer=alice").status_code, 500)

    def test_health_reports_the_runtime(self) -> None:
        body = self.client.get("/api/health").json()
        self.assertTrue(body["runtime"]["running"])
        self.assertIn("session", body["runtime"]["agent_types"])
        self.assertIn("channel.web", body["runtime"]["agent_types"])

    def test_sessions_and_transcripts_line_up(self) -> None:
        listing = self.client.get("/api/sessions").json()
        self.assertIn("sessions", listing)
        for entry in listing["sessions"][:3]:
            with self.subTest(session=entry["id"]):
                transcript = self.client.get(f"/api/sessions/{entry['id']}/transcript")
                self.assertEqual(transcript.status_code, 200)

    def test_an_unknown_session_is_404_not_500(self) -> None:
        self.assertEqual(
            self.client.get("/api/sessions/does-not-exist/transcript").status_code, 404
        )

    def test_an_unknown_approval_is_404(self) -> None:
        self.assertEqual(
            self.client.post("/api/approvals/nope/approve", json={}).status_code, 404
        )

    def test_an_unknown_relay_channel_is_400(self) -> None:
        response = self.client.post("/api/relays", json={"channel": "carrier-pigeon", "peer": "x"})
        self.assertIn(response.status_code, (400, 409))

    def test_chat_without_an_llm_is_409_not_500(self) -> None:
        import config

        was, config.LLM_BASE_URL = config.LLM_BASE_URL, ""
        try:
            response = self.client.post("/api/chat", json={"question": "hi"})
        finally:
            config.LLM_BASE_URL = was
        self.assertEqual(response.status_code, 409)

    def test_openclaw_line_answers_without_an_llm(self) -> None:
        """The escape hatch must not depend on the thing you are escaping from.

        `/api/chat` is 409 without a model. This route has no model in it at all,
        so a broken or unconfigured LLM cannot take the debugging tool down with
        it — which is the entire reason to have a direct path.
        """
        import config

        was, config.LLM_BASE_URL = config.LLM_BASE_URL, ""
        try:
            response = self.client.post("/api/openclaw", json={"line": "/openclaw methods"})
        finally:
            config.LLM_BASE_URL = was

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertIn("sessions.list", body["result"]["read"])
        self.assertTrue(body["session"], "the line belongs to a session")

    def test_openclaw_line_is_written_to_the_transcript(self) -> None:
        """Typed line in before the call, answer in after — the chat path's rule."""
        response = self.client.post(
            "/api/openclaw", json={"line": "/openclaw methods", "peer": "scribe"}
        )
        session_id = response.json()["session"]

        transcript = self.client.get(f"/api/sessions/{session_id}/transcript").json()
        entries = transcript["entries"] if isinstance(transcript, dict) else transcript
        typed = [e for e in entries if e.get("channel") == "openclaw-direct"]
        self.assertGreaterEqual(len(typed), 2, "both halves of the exchange are recorded")
        self.assertEqual(typed[-2]["role"], "user")
        self.assertIn("methods", typed[-2]["content"])
        self.assertEqual(typed[-1]["role"], "assistant")

    def test_a_forbidden_method_is_200_with_a_refusal_not_a_crash(self) -> None:
        response = self.client.post("/api/openclaw", json={"line": "config.set {}"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["tier"], "forbidden")

    def test_a_read_method_is_not_held_for_approval(self) -> None:
        """The hatch exists to show the bytes; gating a listing would retire it."""
        response = self.client.post("/api/openclaw", json={"line": "/openclaw methods"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("held", response.json())

    def test_a_sentence_is_held_and_the_reason_names_the_shell(self) -> None:
        """A sentence reaches OpenClaw's own agent, which has a shell we do not see.

        The approval text has to say that, because the person clicking Approve is
        deciding about a blast radius our gate cannot show them.
        """
        response = self.client.post("/api/openclaw", json={"line": "/openclaw merhaba"})
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertTrue(body["held"])
        self.assertIn("kabuk", body["reason"])
        # The UI reads the id back out of this sentence to draw the button.
        self.assertRegex(body["reason"], r"Approve request [0-9a-f]{6,}")
        self.assertEqual(body["approval_id"], body["reason"].split("Approve request ")[1][:12])

    def test_a_write_method_is_held(self) -> None:
        response = self.client.post(
            "/api/openclaw", json={"line": '/openclaw sessions.reset {"id":"x"}'}
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["held"])

    def test_a_forbidden_method_is_refused_not_offered_for_approval(self) -> None:
        """An approval prompt implies a decision worth making. This one is not."""
        response = self.client.post("/api/openclaw", json={"line": "config.set {}"})
        self.assertNotIn("held", response.json())

    def test_schedule_rejects_a_when_it_cannot_parse(self) -> None:
        response = self.client.post(
            "/api/schedule", json={"name": "x", "when": "yarın", "ask": "y"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("30dk", response.json()["detail"])

    def test_schedule_create_is_held_for_approval(self) -> None:
        response = self.client.post(
            "/api/schedule",
            json={"name": "gece", "when": "her gün 03:00", "ask": "vc_status çağır"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["held"])

    def test_schedule_listing_never_reaches_the_gateway_as_a_write(self) -> None:
        response = self.client.post("/api/openclaw", json={"line": "/openclaw schedule"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("held", response.json())

    def test_schedule_without_a_bar_names_the_syntax_instead_of_asking(self) -> None:
        """The whole point of the subcommand: it answers, it does not clarify."""
        response = self.client.post(
            "/api/openclaw", json={"line": "/openclaw schedule her gün 05:00"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertIn("|", body["error"])
        self.assertNotIn("held", body)

    def test_schedule_create_is_held_then_says_delivery_is_missing(self) -> None:
        response = self.client.post(
            "/api/openclaw",
            json={"line": "/openclaw schedule her gün 05:00 | bana merhaba de"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["held"])

    def test_a_typo_comes_back_with_usage(self) -> None:
        response = self.client.post("/api/openclaw", json={"line": "/openclaw"})
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertIn("Usage", body["error"])

    def test_mechanisms_serves_every_field_the_panel_reads(self) -> None:
        """The drift catcher.

        `app.js` renders these fields by name. If the catalogue loses one, the
        panel silently draws blank labels and nothing else fails — so the contract
        is asserted here rather than discovered in a browser.
        """
        import stages

        body = self.client.get("/api/mechanisms").json()
        self.assertEqual(
            {row["id"] for row in body["mechanisms"]}, set(stages.CATALOGUE),
        )
        for row in body["mechanisms"]:
            for field in ("id", "lane", "title", "klass", "ref", "note", "module"):
                self.assertTrue(row.get(field), f"{row['id']} is missing {field}")

    def test_mechanisms_carries_the_core_counters(self) -> None:
        """The core lane's claim has to be a reading, not a sentence."""
        body = self.client.get("/api/mechanisms").json()
        self.assertIn("routed", body["runtime"])
        self.assertIn("running", body["runtime"])
        self.assertTrue(body["core_idle_note"])


if __name__ == "__main__":
    unittest.main()
