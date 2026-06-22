"""Guardrails — explicit, testable checks run at defined points in the turn.

Each returns `GuardrailResult`s so the eval harness can assert on them directly
(e.g. "the mutation gate fired when the model tried to change a plan without
confirmation"). Three stages:

- input  (pre-LLM)   : jailbreak / prompt-injection detection.
- tool   (pre-exec)  : mutation gate on change_plan.
- output (post-draft): PII masking on the reply text.
"""

from __future__ import annotations

import re

from .events import GuardrailDecision, GuardrailResult
from .state import ConversationState

_JAILBREAK_PATTERNS = [
    r"ignore (all |your )?(previous|prior) instructions",
    r"disregard (the |your )?(system|above)",
    r"reveal (your )?(system )?prompt",
    r"you are now ",
    r"developer mode",
]

_AFFIRMATIVE = re.compile(
    r"\b(yes|yep|yeah|confirm|confirmed|go ahead|do it|please do|sounds good|"
    r"that's right|correct|switch me|change it)\b",
    re.IGNORECASE,
)

# 13–16 consecutive digits (optionally space/dash separated) → card-like.
_CARDISH = re.compile(r"\b(?:\d[ -]?){13,16}\b")

REFUSAL_MESSAGE = (
    "I can only help with Meridian Mobile account and service questions. "
    "Is there something about your account, plan, or service I can help with?"
)


class Guardrails:
    # --- input stage --------------------------------------------------------
    def check_input(self, user_text: str) -> GuardrailResult:
        lowered = user_text.lower()
        for pat in _JAILBREAK_PATTERNS:
            if re.search(pat, lowered):
                return GuardrailResult(
                    name="scope_jailbreak",
                    decision=GuardrailDecision.BLOCK,
                    reason="prompt-injection / out-of-scope instruction detected",
                    stage="pre",
                )
        return GuardrailResult(name="scope_jailbreak", decision=GuardrailDecision.ALLOW, stage="pre")

    def record_confirmations(self, state: ConversationState, user_text: str) -> None:
        """If a plan change is awaiting confirmation and the user affirms, record it.
        Called once per turn before the model runs. Keeps the mutation gate
        deterministic and conversation-driven."""
        if state.pending_confirmation and _AFFIRMATIVE.search(user_text):
            state.confirmed_actions.add(state.pending_confirmation)
            state.pending_confirmation = None

    # --- tool stage ---------------------------------------------------------
    def gate_tool_call(self, state: ConversationState, name: str, args: dict) -> GuardrailResult:
        """Mutation gate. change_plan only runs once the customer has explicitly
        confirmed the specific target plan."""
        if name != "change_plan":
            return GuardrailResult(name="mutation_gate", decision=GuardrailDecision.ALLOW, stage="pre")
        action = f"change_plan:{args.get('plan_id')}"
        if action in state.confirmed_actions:
            return GuardrailResult(name="mutation_gate", decision=GuardrailDecision.ALLOW, stage="pre")
        state.pending_confirmation = action
        return GuardrailResult(
            name="mutation_gate",
            decision=GuardrailDecision.BLOCK,
            reason=f"unconfirmed plan change to {args.get('plan_id')}",
            stage="pre",
        )

    # --- output stage -------------------------------------------------------
    def check_output(self, reply: str) -> tuple[str, GuardrailResult]:
        def _mask(m: re.Match) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            return "•••• " + digits[-4:]

        masked = _CARDISH.sub(_mask, reply)
        if masked != reply:
            return masked, GuardrailResult(
                name="pii_mask",
                decision=GuardrailDecision.MODIFY,
                reason="masked a card-like number in the reply",
                stage="post",
            )
        return reply, GuardrailResult(name="pii_mask", decision=GuardrailDecision.ALLOW, stage="post")
