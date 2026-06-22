"""The Agent core — a stateful, serving-agnostic support agent.

One public method, `turn()`, runs a full user→agent exchange and returns an
`AgentTurn` with the reply plus all structured events (tool calls, retrievals,
guardrail decisions, latency breakdown). The CLI, the eval harness, and the
Baseten serving layer all drive this same object; none of them reach inside it.

Control loop (per turn):
    input guardrail → retrieve KB → [LLM → guardrail tool gate → run tools]* →
    output guardrail → emit AgentTurn
"""

from __future__ import annotations

import json
from typing import Any

from . import prompts, tools as tools_mod
from .events import (
    AgentTurn,
    GuardrailDecision,
    Retrieval,
    Timings,
    ToolCall,
    _Stopwatch,
)
from .guardrails import REFUSAL_MESSAGE, Guardrails
from .llm import LLMClient
from .rag import Retriever
from .state import ConversationState

_MAX_TOOL_ROUNDS = 5  # safety bound on the per-turn tool loop


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever | None = None,
        backend: tools_mod.Backend | None = None,
        guardrails: Guardrails | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever or Retriever()
        self.backend = backend or tools_mod.Backend()
        self.guardrails = guardrails or Guardrails()
        self.state = ConversationState()
        # Persistent Anthropic-shape message log — the model's working memory,
        # including tool_use/tool_result blocks. `state.history` mirrors it as a
        # text-only transcript for the eval harness and CLI.
        self.messages: list[dict[str, Any]] = []

    # -- public API ----------------------------------------------------------
    def turn(self, user_text: str) -> AgentTurn:
        timings = Timings()
        sw = _Stopwatch(timings)
        turn = AgentTurn(reply="", timings=timings)
        self.state.turn_count += 1

        with sw.measure("total_ms"):
            # 1. input guardrail (cheap, pre-LLM)
            with sw.measure("guardrails_ms"):
                gin = self.guardrails.check_input(user_text)
                turn.guardrails.append(gin)
                if gin.decision is GuardrailDecision.BLOCK:
                    turn.reply = REFUSAL_MESSAGE
                    self.state.add("user", user_text)
                    self.state.add("assistant", turn.reply)
                    # don't pollute the model's working log with blocked input
                    return turn
                # promote any pending plan-change confirmation from this message
                self.guardrails.record_confirmations(self.state, user_text)

            # 2. retrieve KB context for this turn
            with sw.measure("retrieval_ms"):
                retrievals = self.retriever.search(user_text, k=3)
            turn.retrievals = retrievals

            # 3. build the user message (KB context prefixed, kept out of system)
            kb_block = prompts.kb_context_block([r.text for r in retrievals])
            user_content = kb_block + user_text if kb_block else user_text
            self.state.add("user", user_text)            # readable transcript
            self.messages.append({"role": "user", "content": user_content})  # model log

            # 4. LLM ↔ tool loop
            final_text = ""
            for _ in range(_MAX_TOOL_ROUNDS):
                with sw.measure("llm_ms"):
                    resp = self.llm.complete(
                        prompts.SYSTEM_PROMPT, self.messages, tools_mod.TOOL_SCHEMAS
                    )
                turn.llm_rounds += 1
                timings.llm_ttft_ms = timings.llm_ttft_ms or resp.ttft_ms

                # echo the assistant turn (incl. tool_use blocks) back verbatim
                self.messages.append({"role": "assistant", "content": resp.raw_assistant_content})

                if not resp.wants_tools():
                    final_text = resp.text()
                    break

                # run each requested tool through the gate, collect tool_results
                tool_results = self._run_tools(resp.tool_uses(), turn, sw)
                self.messages.append({"role": "user", "content": tool_results})
            else:
                # loop exhausted without a final answer
                final_text = final_text or "Let me get a teammate to help with this."

            # 5. output guardrail (PII masking)
            with sw.measure("guardrails_ms"):
                final_text, gout = self.guardrails.check_output(final_text)
                turn.guardrails.append(gout)

            turn.reply = final_text
            turn.escalated = self.state.escalated
            self.state.add("assistant", final_text)

        return turn

    # -- internals -----------------------------------------------------------
    def _run_tools(self, tool_uses, turn: AgentTurn, sw: _Stopwatch) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            gate = self.guardrails.gate_tool_call(self.state, tu.name, tu.input)
            if gate.decision is GuardrailDecision.BLOCK:
                turn.guardrails.append(gate)
                turn.tool_calls.append(
                    ToolCall(name=tu.name, args=tu.input, ok=False, error=gate.reason)
                )
                results.append(self._tool_result(
                    tu.id,
                    {"blocked": True, "reason": gate.reason,
                     "instruction": "Ask the customer to explicitly confirm this exact "
                                    "change before calling this tool again."},
                    is_error=True,
                ))
                continue

            tc = self._execute(tu, sw)
            turn.tool_calls.append(tc)
            if tc.name == "escalate" and tc.ok:
                self.state.escalated = True
            # bind resolved account so later turns/guardrails can key off it
            if tc.name == "lookup_account" and tc.ok and isinstance(tc.result, dict) \
                    and tc.result.get("found"):
                self.state.resolved_account_id = tc.result["account_id"]
            results.append(self._tool_result(tu.id, tc.result, is_error=not tc.ok))
        return results

    def _execute(self, tu, sw: _Stopwatch) -> ToolCall:
        fn = tools_mod.get_tool(tu.name)
        if fn is None:
            return ToolCall(name=tu.name, args=tu.input, ok=False, error="unknown tool")
        try:
            with sw.measure("tools_ms"):
                result = fn(self.backend, **tu.input)
            ok = not (isinstance(result, dict) and "error" in result)
            return ToolCall(name=tu.name, args=tu.input, result=result, ok=ok,
                            error=result.get("error") if not ok else None)
        except TypeError as e:  # bad/missing args from the model
            return ToolCall(name=tu.name, args=tu.input, ok=False, error=f"bad arguments: {e}")

    @staticmethod
    def _tool_result(tool_use_id: str, content: Any, is_error: bool) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps(content),
            "is_error": is_error,
        }
