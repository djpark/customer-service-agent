# Real-time Telco Support Agent — Design

A customer-support agent for a toy **telco** domain, built so the *same core*
serves two outreach angles:

- **Baseten angle** — deploy the inference pipeline (STT → LLM+tools → TTS) on
  Baseten Chains, package with Truss, then benchmark + optimize end-to-end
  latency (target sub-second). → *latency/inference writeup.*
- **Cresta angle** — drive the same agent through a simulation + eval harness:
  scored transcripts, turn- and conversation-level tests, guardrail checks, a
  regression suite with failure-reason reporting. → *agent-reliability writeup.*

Synthetic data only. Measurable results over polish.

---

## 1. The one design decision that makes both angles work

Everything hangs on a **transport- and serving-agnostic Agent core**:

1. **Pluggable LLM backend** behind an `LLMClient` protocol. Dev uses a hosted
   Claude model (best tool-calling, fast to iterate); the Baseten deploy swaps in
   an open model (Llama/Qwen) served via Truss — *no change to agent logic.*
2. **Structured turn events**, not just text. Each `turn()` returns an
   `AgentTurn` carrying the assistant message **plus** the tool calls made,
   retrieved KB chunks, guardrail decisions, and latency breakdown. The CLI
   ignores the extra fields; the eval harness scores them; the latency writeup
   reads the timing fields. One interface, three consumers.

```
                 ┌────────────────────────────┐
   CLI chat ─────┤                            │
   Eval harness ─┤   Agent.turn() -> AgentTurn │──► structured events
   Baseten Chain ┤   (LLMClient injected)      │     (tools, RAG, guardrails,
                 └────────────────────────────┘      latency)
```

If the core leaks its LLM provider or only returns strings, both angles require
rework later. So that's the invariant we protect from milestone 1.

---

## 2. Domain: telco

**Mock backend** (`data/accounts.json`) — synthetic customers, plans, bills,
data usage, line status. Deterministic, seeded, so evals are reproducible.

**Tools** (the agent's only way to touch state):
| Tool | Purpose |
|------|---------|
| `lookup_account(phone_or_id)` | resolve + return account summary |
| `get_billing(account_id)` | current balance, last invoice, due date |
| `get_data_usage(account_id)` | cycle usage vs. plan allowance |
| `check_line_status(account_id)` | active / suspended / outage in area |
| `change_plan(account_id, plan_id)` | mutating — guardrailed, needs confirm |
| `escalate(reason, transcript)` | hand off to human; ends agent control |

**KB** (`kb/*.md`, RAG source) — plan terms & pricing, coverage/outage policy,
troubleshooting (no signal / slow data / SIM), billing policy, cancellation &
returns. Small (~8–12 docs) so retrieval quality is measurable, not luck.

---

## 3. Core agent components

```
support_agent/
  agent/
    core.py        # Agent: stateful session, turn(user_msg) -> AgentTurn
    events.py      # AgentTurn, ToolCall, Retrieval, GuardrailResult, Timings
    llm.py         # LLMClient protocol + ClaudeClient (dev) / BasetenClient
    prompts.py     # system prompt, tool schemas
    tools.py       # tool registry + telco impls over the mock backend
    rag.py         # KB index + retriever (start BM25, embeddings if needed)
    guardrails.py  # pre/post checks
    state.py       # ConversationState (history, resolved account, flags)
  kb/  data/
  cli/chat.py      # interactive REPL (milestone 1 demo)
  eval/            # Cresta angle (milestone 2)
  serving/         # Baseten angle (milestone 3): truss/ + chains/
  writeups/        # milestone 4
```

**Control loop** (per turn): retrieve KB → build prompt → LLM proposes tool
calls → run pre-guardrails → execute tools → loop until final answer →
post-guardrails on the draft → emit `AgentTurn` with full timing breakdown.

**Guardrails** (explicit, testable, logged as `GuardrailResult`):
- *Scope* — refuse off-domain / jailbreak attempts.
- *PII* — never reveal full card/SSN; mask account identifiers.
- *Grounding* — answers about policy must cite a retrieved KB chunk or defer.
- *Mutation gate* — `change_plan` requires explicit user confirmation.
- *Escalation triggers* — repeated failure, billing dispute, churn/legal cues.

**LLM model choice** — dev backend uses a current Claude model (Sonnet for
quality, Haiku for the latency-sensitive path); the Baseten deploy uses an open
model. Final model IDs confirmed against the `claude-api` skill at build time.

---

## 4. Cresta angle — simulation + eval harness (milestone 2)

Reuses the core's structured events; no special hooks in the agent.

- **Personas** (`eval/personas.py`) — seeded synthetic users with a goal, a
  profile in the mock DB, a difficulty/temperament, and hidden facts. A persona
  is itself LLM-driven to produce realistic multi-turn conversation.
- **Simulator** (`eval/simulator.py`) — runs persona ↔ agent to a transcript
  with all structured events captured.
- **Scorers** (`eval/scorers.py`):
  - *turn-level* — correct tool chosen? args valid? guardrail fired when it
    should? grounded claim?
  - *conversation-level* — goal achieved? unnecessary escalation? PII leak?
    turn count vs. budget? LLM-judge rubric for tone/helpfulness.
- **Runner + regression suite** (`eval/runner.py`) — fixed persona set, scored
  each run, **failure-reason reporting** (which check failed, on which turn,
  with the offending event) and a pass/fail gate for regressions.

Deliverable: short reliability writeup — what the harness catches, a real
regression it caught, failure-mode taxonomy.

---

## 5. Baseten angle — serving + latency (milestone 3)

- **Truss** packages the open LLM (and STT/TTS models for the voice path).
- **Chains** composes STT → LLM+tools → TTS as deployed steps; the LLM step
  wraps the same Agent core via `BasetenClient`.
- **Benchmark** — measure end-to-end + per-stage latency (STT, retrieval, LLM
  TTFT/total, tool round-trips, TTS). Optimize toward sub-second: streaming,
  speculative/`Haiku`-class fast path, KV reuse, prompt-cache, tool parallelism.

Deliverable: short latency writeup — stage-by-stage budget, before/after
optimization, what moved the needle.

---

## 6. Milestones

1. **Core agent working** — telco tools + RAG + guardrails + conversation state,
   demoable via CLI chat. Structured `AgentTurn` events from day one.
2. **Eval harness** — personas, simulator, scorers, regression suite + report.
3. **Truss/Chains deploy** + end-to-end latency benchmark & optimization.
4. **Two short writeups** as outreach artifacts.

## 7. Build order for milestone 1

1. `events.py` + `state.py` — the contracts everything else fills in.
2. Mock `data/accounts.json` + `tools.py`.
3. `kb/` docs + `rag.py` retriever.
4. `llm.py` (`ClaudeClient`) + `prompts.py`.
5. `core.py` control loop wiring tools + RAG + guardrails.
6. `guardrails.py`.
7. `cli/chat.py` REPL → first working demo.
