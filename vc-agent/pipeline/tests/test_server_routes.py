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


if __name__ == "__main__":
    unittest.main()
