"""Interactive chat REPL — the milestone-1 demo.

Renders only the reply, but can surface the structured events (tools, retrievals,
guardrails, latency) with --debug, which is a quick window into everything the
eval harness and latency writeup consume.

    python -m support_agent.cli.chat            # needs ANTHROPIC_API_KEY
    python -m support_agent.cli.chat --debug
    python -m support_agent.cli.chat --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import sys

from ..agent.core import Agent
from ..agent.events import GuardrailDecision
from ..agent.llm import ClaudeClient


def _render_debug(turn) -> None:
    t = turn.timings
    print("\n  \033[2m── turn debug ──\033[0m")
    if turn.retrievals:
        kb = ", ".join(f"{r.doc_id}({r.score})" for r in turn.retrievals)
        print(f"  \033[2mkb:      {kb}\033[0m")
    for tc in turn.tool_calls:
        status = "ok" if tc.ok else f"ERR {tc.error}"
        print(f"  \033[2mtool:    {tc.name}({tc.args}) → {status}\033[0m")
    for g in turn.guardrails:
        if g.decision is not GuardrailDecision.ALLOW:
            print(f"  \033[2mguard:   {g.name} {g.decision.value} — {g.reason}\033[0m")
    print(
        f"  \033[2mlatency: {t.total_ms:.0f}ms total "
        f"(llm {t.llm_ms:.0f}, ttft {t.llm_ttft_ms:.0f}, "
        f"rag {t.retrieval_ms:.1f}, tools {t.tools_ms:.0f}) "
        f"· {turn.llm_rounds} llm round(s)\033[0m\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Meridian Mobile support agent — chat")
    ap.add_argument("--debug", action="store_true", help="show tools/guardrails/latency")
    ap.add_argument("--model", default=None, help="override the LLM model id")
    ap.add_argument("--thinking", action="store_true", help="enable adaptive thinking")
    args = ap.parse_args()

    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model
    if args.thinking:
        client_kwargs["thinking"] = True

    try:
        agent = Agent(llm=ClaudeClient(**client_kwargs))
    except Exception as e:  # most often a missing ANTHROPIC_API_KEY / missing dep
        print(f"Could not start the agent: {e}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY and `pip install anthropic`.", file=sys.stderr)
        return 1

    print("Meridian Mobile support — type 'quit' to exit."
          + ("  [debug on]" if args.debug else ""))
    print("Ada: Hi, I'm Ada. Can I get your phone number or account id to start?")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        turn = agent.turn(user)
        print(f"Ada: {turn.reply}")
        if args.debug:
            _render_debug(turn)
        if turn.escalated:
            print("  \033[2m(conversation handed off to a human agent)\033[0m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
