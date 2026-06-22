"""Structured turn events — the contract shared by the CLI, the eval harness,
and the Baseten serving layer.

The CLI renders only `AgentTurn.reply`. The eval harness scores `tool_calls`,
`retrievals`, and `guardrails`. The latency writeup reads `timings`. Keeping all
of this on one object is what lets a single Agent core serve all three consumers
without provider- or transport-specific branches.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"  # answer was rewritten/redacted before returning


@dataclass
class ToolCall:
    """One tool invocation the agent made during a turn."""

    name: str
    args: dict[str, Any]
    result: Any = None
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class Retrieval:
    """A KB chunk surfaced by RAG for this turn."""

    doc_id: str
    score: float
    text: str


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail check. The eval harness asserts on these."""

    name: str
    decision: GuardrailDecision
    reason: str = ""
    stage: str = "pre"  # "pre" (before tools) or "post" (on the draft reply)


@dataclass
class Timings:
    """Per-stage latency breakdown in milliseconds. Read by the latency writeup.

    `total_ms` is wall-clock for the whole turn; the rest are component spans and
    may overlap or sum to less than the total (e.g. queueing, glue code).
    """

    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # time to first token, if the backend streams
    tools_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class AgentTurn:
    """Everything that happened in one user->agent exchange."""

    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrievals: list[Retrieval] = field(default_factory=list)
    guardrails: list[GuardrailResult] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    escalated: bool = False
    # raw LLM rounds (for debugging / eval introspection); not rendered to users
    llm_rounds: int = 0

    def blocked(self) -> bool:
        return any(g.decision is GuardrailDecision.BLOCK for g in self.guardrails)

    def used_tool(self, name: str) -> bool:
        return any(tc.name == name for tc in self.tool_calls)


class _Stopwatch:
    """Accumulates elapsed time into a Timings field via a context manager.

    Usage:
        sw = _Stopwatch(timings)
        with sw.measure("retrieval_ms"):
            ...
    """

    def __init__(self, timings: Timings) -> None:
        self.timings = timings

    @contextmanager
    def measure(self, field_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            setattr(self.timings, field_name,
                    getattr(self.timings, field_name) + elapsed)
