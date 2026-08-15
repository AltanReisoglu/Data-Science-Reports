"""The context engine: token budget, and compaction that never splits a tool block."""

from __future__ import annotations

import unittest

from autogen_core import FunctionCall
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    UserMessage,
)

import context_engine


def user(text: str) -> UserMessage:
    return UserMessage(content=text, source="operator")


def reply(text: str) -> AssistantMessage:
    return AssistantMessage(content=text, source="Analyst")


def tool_call(call_id: str, name: str = "search_docs") -> AssistantMessage:
    return AssistantMessage(
        content=[FunctionCall(id=call_id, name=name, arguments="{}")], source="Analyst"
    )


def tool_result(call_id: str, name: str = "search_docs") -> FunctionExecutionResultMessage:
    return FunctionExecutionResultMessage(
        content=[
            FunctionExecutionResult(call_id=call_id, content="result", is_error=False, name=name)
        ]
    )


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_counting_is_by_size_not_by_message_count(self) -> None:
        """The flaw the engine exists to fix: 24 short turns != 24 long ones."""
        short = [user("hi") for _ in range(20)]
        long = [user("x" * 4000) for _ in range(20)]
        self.assertLess(
            context_engine.estimate_tokens(short),
            context_engine.estimate_tokens(long) // 10,
        )

    async def test_a_conversation_inside_the_budget_is_left_alone(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(token_budget=10_000, reserve=1_000)
        for i in range(6):
            await ctx.add_message(user(f"question {i}"))

        messages = await ctx.get_messages()
        self.assertEqual(len(messages), 6)
        self.assertEqual(ctx.stats.compacted, 0)

    async def test_going_over_budget_compacts_on_assemble(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=600, reserve=100, keep_recent=2
        )
        for i in range(12):
            await ctx.add_message(user(f"question {i} " + "padding " * 30))

        messages = await ctx.get_messages()
        self.assertEqual(ctx.stats.compacted, 1)
        self.assertLess(len(messages), 12)
        # The summary is prepended so nothing simply vanishes without a trace.
        self.assertTrue(str(messages[0].content).startswith(context_engine.SUMMARY_PREFIX))

    async def test_recent_turns_survive_compaction(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=500, reserve=50, keep_recent=3
        )
        for i in range(10):
            await ctx.add_message(user(f"msg-{i} " + "padding " * 30))

        rendered = " ".join(str(m.content) for m in await ctx.get_messages())
        self.assertIn("msg-9", rendered)
        self.assertNotIn("msg-0 ", rendered)


class ToolPairingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cut_landing_on_a_tool_result_is_moved_past_it(self) -> None:
        """A result whose call was summarised away is a sequence providers reject."""
        messages = [user("a"), tool_call("1"), tool_result("1"), reply("b")]
        self.assertEqual(context_engine._safe_boundary(messages, 2), 3)

    async def test_a_cut_landing_between_call_and_result_is_moved(self) -> None:
        messages = [tool_call("1"), tool_result("1"), user("next")]
        self.assertEqual(context_engine._safe_boundary(messages, 1), 2)

    async def test_a_safe_cut_is_left_where_it_is(self) -> None:
        messages = [user("a"), reply("b"), user("c")]
        self.assertEqual(context_engine._safe_boundary(messages, 2), 2)

    async def test_compaction_never_leaves_a_dangling_tool_result(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2
        )
        for i in range(6):
            await ctx.add_message(user(f"ask {i} " + "padding " * 20))
            await ctx.add_message(tool_call(str(i)))
            await ctx.add_message(tool_result(str(i)))
            await ctx.add_message(reply(f"answer {i} " + "padding " * 20))

        messages = await ctx.get_messages()
        body = [m for m in messages if not str(getattr(m, "content", "")).startswith(
            context_engine.SUMMARY_PREFIX
        )]

        # Every surviving result must have its call immediately before it.
        for index, message in enumerate(body):
            if context_engine._is_tool_result(message):
                self.assertGreater(index, 0, "a result opened the context with no call")
                self.assertTrue(
                    context_engine._is_tool_call(body[index - 1]),
                    "a tool result survived without its call",
                )


class SummariserTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_summariser_is_used_when_it_works(self) -> None:
        async def summarise(messages):
            return f"{len(messages)} turns about fintech"

        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2, summariser=summarise
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))

        rendered = str((await ctx.get_messages())[0].content)
        self.assertIn("turns about fintech", rendered)
        self.assertEqual(ctx.stats.compactions[-1].method, "model")

    async def test_a_broken_summariser_falls_back_instead_of_raising(self) -> None:
        """A dead endpoint must not take the conversation down with it."""

        async def broken(messages):
            raise RuntimeError("endpoint down")

        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2, summariser=broken
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))

        messages = await ctx.get_messages()
        self.assertTrue(messages)
        event = ctx.stats.compactions[-1]
        self.assertEqual(event.method, "truncate")
        self.assertIn("endpoint down", event.note)

    async def test_the_fallback_admits_what_it_dropped(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))

        summary = str((await ctx.get_messages())[0].content)
        self.assertIn("were dropped", summary)
        self.assertIn("not summarised", summary)
        self.assertIn("ask again", summary)

    async def test_an_empty_summary_is_treated_as_a_failure(self) -> None:
        async def empty(messages):
            return "   "

        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2, summariser=empty
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))
        await ctx.get_messages()
        self.assertEqual(ctx.stats.compactions[-1].method, "truncate")


class StateTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_summary_survives_save_and_load(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))
        await ctx.get_messages()

        restored = context_engine.CompactingChatCompletionContext()
        await restored.load_state(await ctx.save_state())

        rendered = str((await restored.get_messages())[0].content)
        self.assertTrue(rendered.startswith(context_engine.SUMMARY_PREFIX))

    async def test_clear_drops_the_summary_too(self) -> None:
        ctx = context_engine.CompactingChatCompletionContext(
            token_budget=400, reserve=50, keep_recent=2
        )
        for i in range(10):
            await ctx.add_message(user(f"m{i} " + "padding " * 30))
        await ctx.get_messages()

        await ctx.clear()
        self.assertEqual(await ctx.get_messages(), [])


if __name__ == "__main__":
    unittest.main()
