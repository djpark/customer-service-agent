"""LLM backend abstraction.

The Agent core depends only on the `LLMClient` protocol and the normalized
`LLMResponse` it returns — never on a concrete provider. The dev/default backend
is `ClaudeClient` (hosted Claude, best tool-calling). The Baseten deploy will add
a `BasetenClient` that serves an open model (Llama/Qwen) behind the same protocol,
with no change to `core.py`. The eval harness can inject a scripted stub the same
way.

Messages and tool schemas use the Anthropic Messages wire shape as the lingua
franca (clean, well-specified); a non-Anthropic client implements the same
protocol by translating that shape on the way in/out.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# Default dev model. Opus 4.8 for best tool-use quality; swap to a Haiku-class
# model for the latency-sensitive path (the Baseten benchmark explores this).
DEFAULT_MODEL = "claude-opus-4-8"
LATENCY_MODEL = "claude-haiku-4-5"


@dataclass
class TextPart:
    text: str


@dataclass
class ToolUsePart:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Provider-neutral result of one model call."""

    parts: list[Any] = field(default_factory=list)  # TextPart | ToolUsePart
    stop_reason: str = "end_turn"
    raw_assistant_content: Any = None  # echo back verbatim into message history
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0

    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart)).strip()

    def tool_uses(self) -> list[ToolUsePart]:
        return [p for p in self.parts if isinstance(p, ToolUsePart)]

    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use"


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class ClaudeClient:
    """Hosted-Claude backend. Streams the call to capture time-to-first-token for
    the latency writeup, then assembles the full message via the SDK helper.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        effort: str | None = "low",   # real-time support → snappy turns
        thinking: bool = False,       # off by default for latency; on for hard cases
        api_key: str | None = None,
    ) -> None:
        import anthropic  # lazy: keeps the package importable without the dep

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system, messages, tools) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "tools": tools,
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        start = time.perf_counter()
        ttft: float | None = None
        with self._client.messages.stream(**kwargs) as stream:
            for _ in stream.text_stream:
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000.0
            message = stream.get_final_message()
        total_ms = (time.perf_counter() - start) * 1000.0

        parts: list[Any] = []
        for block in message.content:
            if block.type == "text":
                parts.append(TextPart(text=block.text))
            elif block.type == "tool_use":
                parts.append(ToolUsePart(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            parts=parts,
            stop_reason=message.stop_reason or "end_turn",
            raw_assistant_content=message.content,  # echoed back into history verbatim
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            ttft_ms=ttft or total_ms,
            total_ms=total_ms,
        )
