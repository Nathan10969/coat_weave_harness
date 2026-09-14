"""Streaming tool-calling runtime for OpenAI-compatible clients."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


MAX_TOOL_OUTPUT_CHARS = 60_000


class AgentRuntimeError(RuntimeError):
    pass


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class AgentEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "timestamp": self.timestamp, **self.data}


@dataclass
class AgentRunResult:
    answer: str
    messages: List[Dict[str, Any]]
    tool_trace: List[Dict[str, Any]]
    tool_call_count: int
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "messages": self.messages,
            "tool_trace": self.tool_trace,
            "tool_call_count": self.tool_call_count,
            "finish_reason": self.finish_reason,
        }


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[..., str]

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Tool schemas paired with local execute callbacks."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"Tool name is missing or already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str | Dict[str, Any]) -> str:
        if name not in self._tools:
            raise ValueError(f"Unknown registered tool: {name}")
        params = (
            json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        )
        if not isinstance(params, dict):
            raise ValueError("Tool arguments must decode to a JSON object")
        result = self._tools[name].execute(**params)
        output = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            return output[:MAX_TOOL_OUTPUT_CHARS] + "\n[TOOL_OUTPUT_TRUNCATED]"
        return output


class ToolCallingAgent:
    """Streaming function-calling loop with retries and structured events."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        registry: ToolRegistry,
        system_prompt: str,
        max_rounds: int = 24,
        max_tool_calls: int = 80,
        max_total_tool_output_chars: int = 300_000,
        request_retries: int = 3,
        tool_retries: int = 2,
        retry_base_seconds: float = 1.0,
        require_tool_use: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_rounds = max(1, max_rounds)
        self.max_tool_calls = max(1, max_tool_calls)
        self.max_total_tool_output_chars = max(1, max_total_tool_output_chars)
        self.request_retries = max(0, request_retries)
        self.tool_retries = max(0, tool_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.require_tool_use = require_tool_use
        self.model_kwargs = dict(model_kwargs or {})

    def _delay(self, attempt: int) -> float:
        base = self.retry_base_seconds * (2 ** max(0, attempt - 1))
        return base + random.uniform(0, min(0.25, base / 4 if base else 0))

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _open_stream(
        self, messages: List[Dict[str, Any]], tool_choice: str
    ) -> Iterator[AgentEvent]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.request_retries + 2):
            request = {
                "model": self.model,
                "messages": messages,
                **self.model_kwargs,
                "tools": self.registry.schemas(),
                "tool_choice": tool_choice,
                "stream": True,
            }
            try:
                stream = self.client.chat.completions.create(**request)
                return stream
            except Exception as exc:
                last_error = exc
                if attempt > self.request_retries:
                    break
                delay = self._delay(attempt)
                yield AgentEvent(
                    "retry",
                    {
                        "scope": "api",
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._wait(delay)
        raise AgentRuntimeError(f"API request failed after retries: {last_error}")

    def _consume_stream(self, stream: Any) -> Iterator[AgentEvent]:
        content_parts: List[str] = []
        tool_parts: Dict[int, Dict[str, str]] = {}
        finish_reason: Optional[str] = None
        for chunk in stream:
            choices = _value(chunk, "choices", []) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = _value(choice, "finish_reason", finish_reason)
            delta = _value(choice, "delta", {}) or {}
            content = _value(delta, "content")
            if content:
                content_parts.append(content)
                yield AgentEvent("assistant_delta", {"text": content})
            for call_delta in _value(delta, "tool_calls", []) or []:
                index = int(_value(call_delta, "index", 0) or 0)
                state = tool_parts.setdefault(
                    index, {"id": "", "name": "", "arguments": "", "type": "function"}
                )
                call_id = _value(call_delta, "id")
                if call_id:
                    state["id"] = call_id
                call_type = _value(call_delta, "type")
                if call_type:
                    state["type"] = call_type
                function = _value(call_delta, "function", {}) or {}
                name = _value(function, "name")
                arguments = _value(function, "arguments")
                if name:
                    state["name"] += name
                if arguments:
                    state["arguments"] += arguments
        calls = [
            {
                "id": state["id"] or f"tool-{index}",
                "type": state["type"],
                "function": {
                    "name": state["name"],
                    "arguments": state["arguments"] or "{}",
                },
            }
            for index, state in sorted(tool_parts.items())
        ]
        return "".join(content_parts), calls, finish_reason

    def _execute_tool(self, name: str, arguments: str) -> Iterator[AgentEvent]:
        last_error: Optional[Exception] = None
        started = time.monotonic()
        for attempt in range(1, self.tool_retries + 2):
            try:
                output = self.registry.execute(name, arguments)
                return output, attempt, time.monotonic() - started
            except Exception as exc:
                last_error = exc
                if attempt > self.tool_retries:
                    break
                delay = self._delay(attempt)
                yield AgentEvent(
                    "retry",
                    {
                        "scope": "tool",
                        "tool": name,
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._wait(delay)
        output = f"TOOL_ERROR: {type(last_error).__name__}: {last_error}"
        return output, self.tool_retries + 1, time.monotonic() - started

    def stream(self, user_message: str) -> Iterator[AgentEvent]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_trace: List[Dict[str, Any]] = []
        tool_call_count = 0
        total_tool_output_chars = 0
        no_tool_retry_used = False

        for round_index in range(1, self.max_rounds + 1):
            tool_choice = "auto"
            yield AgentEvent(
                "round_start", {"round": round_index, "tool_choice": tool_choice}
            )
            response_stream = yield from self._open_stream(messages, tool_choice)
            content, requested_calls, finish_reason = yield from self._consume_stream(
                response_stream
            )
            if not requested_calls:
                if (
                    self.require_tool_use
                    and tool_call_count == 0
                    and not no_tool_retry_used
                ):
                    no_tool_retry_used = True
                    messages.append({"role": "assistant", "content": content or ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": "You must call read on manifest.json and use the sandbox tools before answering.",
                        }
                    )
                    yield AgentEvent(
                        "warning",
                        {
                            "message": "Model returned no tool call; requesting tool use again."
                        },
                    )
                    continue
                if self.require_tool_use and tool_call_count == 0:
                    raise AgentRuntimeError(
                        "Model did not use the required patent tools"
                    )
                messages.append({"role": "assistant", "content": content})
                result = AgentRunResult(
                    answer=content or "",
                    messages=messages,
                    tool_trace=tool_trace,
                    tool_call_count=tool_call_count,
                    finish_reason=finish_reason,
                )
                yield AgentEvent("final", {"result": result})
                return

            remaining_calls = self.max_tool_calls - tool_call_count
            requested_calls = requested_calls[:remaining_calls]
            if not requested_calls:
                break
            assistant_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": requested_calls,
            }
            messages.append(assistant_message)
            for call in requested_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                tool_call_count += 1
                yield AgentEvent(
                    "tool_start",
                    {
                        "tool_call_id": call["id"],
                        "name": name,
                        "arguments": arguments,
                    },
                )
                remaining_output_chars = (
                    self.max_total_tool_output_chars - total_tool_output_chars
                )
                if remaining_output_chars <= 0:
                    output = "TOOL_ERROR: cumulative tool-output budget exhausted"
                    attempts = 0
                    duration = 0.0
                else:
                    output, attempts, duration = yield from self._execute_tool(
                        name, arguments
                    )
                    if len(output) > remaining_output_chars:
                        marker = "\n[TOOL_OUTPUT_BUDGET_EXHAUSTED]"
                        prefix_length = remaining_output_chars - len(marker)
                        output = (
                            output[:prefix_length] + marker
                            if prefix_length > 0
                            else output[:remaining_output_chars]
                        )
                    total_tool_output_chars += len(output)
                trace_item = {
                    "tool_call_id": call["id"],
                    "name": name,
                    "arguments": arguments,
                    "output": output,
                    "attempts": attempts,
                    "duration_seconds": duration,
                    "total_tool_output_chars": total_tool_output_chars,
                }
                tool_trace.append(trace_item)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": output,
                    }
                )
                yield AgentEvent("tool_end", trace_item)
            if (
                tool_call_count >= self.max_tool_calls
                or total_tool_output_chars >= self.max_total_tool_output_chars
            ):
                break

        if self.require_tool_use and tool_call_count == 0:
            raise AgentRuntimeError("Model did not use the required patent tools")
        messages.append(
            {
                "role": "user",
                "content": "Tool/round budget is complete. Return the final evidence-backed answer now.",
            }
        )
        yield AgentEvent(
            "round_start", {"round": self.max_rounds + 1, "tool_choice": "none"}
        )
        response_stream = yield from self._open_stream(messages, "none")
        content, _, finish_reason = yield from self._consume_stream(response_stream)
        messages.append({"role": "assistant", "content": content})
        result = AgentRunResult(
            answer=content or "",
            messages=messages,
            tool_trace=tool_trace,
            tool_call_count=tool_call_count,
            finish_reason=finish_reason or "budget",
        )
        yield AgentEvent("final", {"result": result})
