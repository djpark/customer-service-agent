# Meridian Mobile — Real-time Telco Support Agent

A customer-support agent for a toy telecom domain, built so one core serves two
goals: a **latency/inference** track (deploy on Baseten, benchmark end-to-end)
and a **reliability/eval** track (simulate personas, score transcripts, catch
regressions). Synthetic data only.

See [DESIGN.md](DESIGN.md) for the architecture and the rationale behind the two
tracks.

## Status

- [x] **Milestone 1 — core agent.** Telco tools + RAG + guardrails + conversation
      state, driven by a provider-agnostic LLM backend, demoable via CLI chat.
      Every turn emits a structured `AgentTurn` (tools, retrievals, guardrails,
      latency) — the contract the other two tracks consume.
- [ ] Milestone 2 — eval harness (personas, simulator, scorers, regression suite)
- [ ] Milestone 3 — Truss/Chains deploy + latency benchmark
- [ ] Milestone 4 — two short writeups

## Architecture in one breath

```
   CLI chat ─────┐
   Eval harness ─┤──►  Agent.turn(user) ──► AgentTurn   (reply + structured events)
   Baseten Chain ┘     (LLMClient injected)
```

The `Agent` core depends only on an `LLMClient` protocol and returns structured
turn events. The CLI renders `reply`; the eval harness scores the events; the
latency writeup reads the timings. Swapping the hosted Claude backend for an
open model on Baseten is a client swap, not an agent change.

## Layout

```
support_agent/
  agent/
    core.py        # Agent: control loop → AgentTurn
    events.py      # AgentTurn / ToolCall / Retrieval / GuardrailResult / Timings
    state.py       # ConversationState
    llm.py         # LLMClient protocol + ClaudeClient (dev backend)
    prompts.py     # system prompt + KB context injection
    tools.py       # telco tools over a seeded mock backend (data/accounts.json)
    rag.py         # BM25 retriever over kb/*.md
    guardrails.py  # jailbreak / mutation-gate / PII-mask checks
  kb/              # telco knowledge base (plans, billing, coverage, troubleshooting, cancellation)
  data/            # synthetic accounts
  cli/chat.py      # interactive REPL (milestone-1 demo)
tests/smoke.py     # no-API verification of tools, RAG, guardrails, control loop
```

## Run

No-API smoke test (verifies tools, RAG, guardrails, and the full control loop
with a scripted stub LLM — no key needed):

```bash
python -m tests.smoke
```

Live chat demo (needs a key):

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python -m support_agent.cli.chat --debug          # --debug shows tools/guardrails/latency
python -m support_agent.cli.chat --model claude-haiku-4-5   # latency-path model
```

Try: `hi, what do I owe? my number is 415-555-0101` · `why is my data slow?` ·
`switch me to premium unlimited` (watch the mutation gate ask for confirmation).

## Notes

- **Dev model** defaults to `claude-opus-4-8`; the latency track swaps to a
  Haiku-class model. Thinking is off and effort is `low` by default — this is a
  real-time agent — both are `ClaudeClient` constructor knobs.
- **RAG** is BM25 (zero infra, reproducible) so retrieval quality is measurable;
  it moves to embeddings only if quality demands it.
