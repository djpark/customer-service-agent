"""Conversation state carried across turns within one session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: Any  # str for user/assistant text; structured blocks otherwise


@dataclass
class ConversationState:
    """Per-session memory. Owned by the Agent, inspected by the eval harness.

    `resolved_account_id` is the one piece of cross-turn business state that
    matters most: once the agent has identified the caller, downstream tools and
    guardrails (PII masking, mutation gate) key off it.
    """

    history: list[Message] = field(default_factory=list)
    resolved_account_id: str | None = None
    # mutating actions the user has explicitly confirmed this session, e.g.
    # {"change_plan:premium_unlimited"} — consumed by the mutation gate.
    confirmed_actions: set[str] = field(default_factory=set)
    # a mutating action the agent proposed and is awaiting confirmation for, e.g.
    # "change_plan:premium_unlimited". Set by the mutation gate on block; promoted
    # into confirmed_actions when the user affirms next turn.
    pending_confirmation: str | None = None
    escalated: bool = False
    turn_count: int = 0

    def add(self, role: Role, content: Any) -> None:
        self.history.append(Message(role=role, content=content))

    def last_user_text(self) -> str:
        for msg in reversed(self.history):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return ""
