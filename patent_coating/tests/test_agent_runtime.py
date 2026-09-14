from __future__ import annotations

import copy
import types
import unittest

from patent_coating.search.agent_runtime import (
    AgentRuntimeError,
    RegisteredTool,
    ToolCallingAgent,
    ToolRegistry,
)


def stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


class FakeCompletions:
    def __init__(self, responses: list[list[dict[str, object]]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected extra OpenAI request")
        return iter(self.responses.pop(0))


class FakeOpenAIClient:
    def __init__(self, responses: list[list[dict[str, object]]]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = types.SimpleNamespace(completions=self.completions)


def make_registry(recorded_values: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    def echo(value: str) -> str:
        recorded_values.append(value)
        return f"TOOL SAW: {value}"

    registry.register(
        RegisteredTool(
            name="echo",
            description="Echo a value",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            execute=echo,
        )
    )
    return registry


class ToolCallingAgentTests(unittest.TestCase):
    def test_fake_openai_stream_executes_fragmented_tool_call_then_finishes(
        self,
    ) -> None:
        first_response = [
            stream_chunk(
                content="Checking. ",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "ec", "arguments": '{"value":'},
                    }
                ],
            ),
            stream_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "ho", "arguments": ' "evidence"}'},
                    }
                ],
                finish_reason="tool_calls",
            ),
        ]
        second_response = [
            stream_chunk(content="Supported "),
            stream_chunk(content="answer.", finish_reason="stop"),
        ]
        client = FakeOpenAIClient([first_response, second_response])
        recorded_values: list[str] = []
        agent = ToolCallingAgent(
            client=client,
            model="fake-model",
            registry=make_registry(recorded_values),
            system_prompt="Use evidence tools.",
            max_rounds=3,
            retry_base_seconds=0,
            require_tool_use=True,
        )

        events = list(agent.stream("Research the patent"))

        self.assertEqual(recorded_values, ["evidence"])
        self.assertEqual(len(client.completions.calls), 2)
        self.assertEqual(client.completions.calls[0]["model"], "fake-model")
        self.assertEqual(client.completions.calls[0]["tool_choice"], "auto")
        self.assertIs(client.completions.calls[0]["stream"], True)
        self.assertEqual(
            client.completions.calls[0]["tools"][0]["function"]["name"],
            "echo",
        )
        second_messages = client.completions.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call-1")
        self.assertEqual(second_messages[-1]["content"], "TOOL SAW: evidence")

        kinds = [event.kind for event in events]
        self.assertEqual(kinds.count("tool_start"), 1)
        self.assertEqual(kinds.count("tool_end"), 1)
        self.assertEqual(kinds[-1], "final")
        result = events[-1].data["result"]
        self.assertEqual(result.answer, "Supported answer.")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_trace[0]["name"], "echo")
        self.assertEqual(result.tool_trace[0]["attempts"], 1)

    def test_required_tool_policy_reprompts_once_then_rejects_no_tool_answer(
        self,
    ) -> None:
        client = FakeOpenAIClient(
            [
                [stream_chunk(content="Premature answer", finish_reason="stop")],
                [stream_chunk(content="Still no tool", finish_reason="stop")],
            ]
        )
        agent = ToolCallingAgent(
            client=client,
            model="fake-model",
            registry=make_registry([]),
            system_prompt="Use tools.",
            max_rounds=3,
            request_retries=0,
            retry_base_seconds=0,
            require_tool_use=True,
        )

        emitted = []
        with self.assertRaisesRegex(
            AgentRuntimeError, "did not use the required patent tools"
        ):
            for event in agent.stream("question"):
                emitted.append(event)

        self.assertIn("warning", [event.kind for event in emitted])
        self.assertEqual(len(client.completions.calls), 2)
        retry_messages = client.completions.calls[1]["messages"]
        self.assertEqual(retry_messages[-1]["role"], "user")
        self.assertIn("must call read on manifest.json", retry_messages[-1]["content"])

    def test_required_tool_policy_cannot_fall_through_round_budget(self) -> None:
        client = FakeOpenAIClient(
            [[stream_chunk(content="No tool", finish_reason="stop")]]
        )
        agent = ToolCallingAgent(
            client=client,
            model="fake-model",
            registry=make_registry([]),
            system_prompt="Use tools.",
            max_rounds=1,
            request_retries=0,
            retry_base_seconds=0,
            require_tool_use=True,
        )

        with self.assertRaisesRegex(
            AgentRuntimeError, "did not use the required patent tools"
        ):
            list(agent.stream("question"))
        self.assertEqual(len(client.completions.calls), 1)

    def test_cumulative_tool_output_budget_stops_more_tool_execution(self) -> None:
        tool_calls = [
            {
                "index": index,
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json_arguments,
                },
            }
            for index, json_arguments in enumerate(
                ('{"value":"first evidence"}', '{"value":"second evidence"}')
            )
        ]
        client = FakeOpenAIClient(
            [
                [stream_chunk(tool_calls=tool_calls, finish_reason="tool_calls")],
                [stream_chunk(content="Budgeted answer", finish_reason="stop")],
            ]
        )
        recorded_values: list[str] = []
        agent = ToolCallingAgent(
            client=client,
            model="fake-model",
            registry=make_registry(recorded_values),
            system_prompt="Use tools.",
            max_rounds=3,
            max_total_tool_output_chars=12,
            request_retries=0,
            retry_base_seconds=0,
            require_tool_use=True,
        )

        events = list(agent.stream("question"))

        self.assertEqual(recorded_values, ["first evidence"])
        result = events[-1].data["result"]
        self.assertEqual(result.answer, "Budgeted answer")
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(len(result.tool_trace[0]["output"]), 12)
        self.assertEqual(
            result.tool_trace[1]["output"],
            "TOOL_ERROR: cumulative tool-output budget exhausted",
        )
        self.assertEqual(result.tool_trace[1]["attempts"], 0)
        self.assertEqual(client.completions.calls[-1]["tool_choice"], "none")


if __name__ == "__main__":
    unittest.main()
