"""Telco tools — the agent's only channel to mutate or read account state.

Each tool is a plain function over an in-memory mock backend loaded from
`data/accounts.json`. The backend is deterministic and seeded so eval runs are
reproducible. Tools never print; they return JSON-able dicts that become
`ToolCall.result` in the structured turn.

The `TOOL_SCHEMAS` list is the provider-neutral description of each tool. The
LLMClient adapts it to whatever the backing model expects.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "accounts.json"


def _mask_phone(phone: str) -> str:
    return phone[:-4].replace("+1", "") + "••" + phone[-2:] if phone else phone


class Backend:
    """Mutable in-memory store. A fresh instance == a fresh world (good for evals)."""

    def __init__(self, data: dict | None = None) -> None:
        raw = data if data is not None else json.loads(_DATA_PATH.read_text())
        self._data = copy.deepcopy(raw)
        self._by_id = {a["account_id"]: a for a in self._data["accounts"]}
        self._by_phone = {a["phone"]: a for a in self._data["accounts"]}

    # --- resolution helpers -------------------------------------------------
    def _find(self, phone_or_id: str) -> dict | None:
        key = (phone_or_id or "").strip()
        if key in self._by_id:
            return self._by_id[key]
        # tolerate phone numbers with/without formatting
        digits = "".join(c for c in key if c.isdigit())
        for phone, acct in self._by_phone.items():
            if "".join(c for c in phone if c.isdigit()).endswith(digits) and digits:
                return acct
        return None

    @property
    def plans(self) -> dict:
        return self._data["plans"]


# --- tool implementations ---------------------------------------------------
# Each takes the Backend as first arg (bound by the registry) plus named args
# the model supplies. Returns a JSON-able dict. Errors are returned, not raised,
# so the agent can recover conversationally.

def lookup_account(backend: Backend, phone_or_id: str) -> dict[str, Any]:
    acct = backend._find(phone_or_id)
    if not acct:
        return {"found": False, "error": "no account matches that phone or id"}
    plan = backend.plans[acct["plan_id"]]
    return {
        "found": True,
        "account_id": acct["account_id"],
        "name": acct["name"],
        "phone_masked": _mask_phone(acct["phone"]),
        "zip": acct["zip"],
        "plan": {"id": acct["plan_id"], "name": plan["name"], "price_monthly": plan["price_monthly"]},
        "line_status": acct["line_status"],
        "card_last4": acct["card_last4"],
    }


def get_billing(backend: Backend, account_id: str) -> dict[str, Any]:
    acct = backend._by_id.get(account_id)
    if not acct:
        return {"error": "unknown account_id"}
    return {"account_id": account_id, **acct["billing"]}


def get_data_usage(backend: Backend, account_id: str) -> dict[str, Any]:
    acct = backend._by_id.get(account_id)
    if not acct:
        return {"error": "unknown account_id"}
    plan = backend.plans[acct["plan_id"]]
    allowance = plan["data_gb"]
    used = acct["usage"]["data_used_gb"]
    return {
        "account_id": account_id,
        "cycle_start": acct["usage"]["cycle_start"],
        "data_used_gb": used,
        "allowance_gb": allowance,  # null == unlimited
        "over_allowance": (allowance is not None and used > allowance),
    }


def check_line_status(backend: Backend, account_id: str) -> dict[str, Any]:
    acct = backend._by_id.get(account_id)
    if not acct:
        return {"error": "unknown account_id"}
    outage = next((o for o in backend._data["outages"] if o["area"] == acct["zip"]), None)
    return {
        "account_id": account_id,
        "line_status": acct["line_status"],
        "area_outage": outage,  # null if none
    }


def change_plan(backend: Backend, account_id: str, plan_id: str) -> dict[str, Any]:
    """Mutating. The mutation-gate guardrail must have confirmed before this runs."""
    acct = backend._by_id.get(account_id)
    if not acct:
        return {"error": "unknown account_id"}
    if plan_id not in backend.plans:
        return {"error": f"unknown plan_id: {plan_id}", "valid_plans": list(backend.plans)}
    old = acct["plan_id"]
    acct["plan_id"] = plan_id
    return {
        "account_id": account_id,
        "changed": True,
        "from_plan": old,
        "to_plan": plan_id,
        "new_price_monthly": backend.plans[plan_id]["price_monthly"],
    }


def escalate(backend: Backend, reason: str, summary: str = "") -> dict[str, Any]:
    """Hand off to a human. Ends agent control of the conversation."""
    return {"escalated": True, "reason": reason, "summary": summary}


# --- registry ---------------------------------------------------------------

ToolFn = Callable[..., dict[str, Any]]

_REGISTRY: dict[str, ToolFn] = {
    "lookup_account": lookup_account,
    "get_billing": get_billing,
    "get_data_usage": get_data_usage,
    "check_line_status": check_line_status,
    "change_plan": change_plan,
    "escalate": escalate,
}

# Provider-neutral tool schemas. Mirrors the function signatures above.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_account",
        "description": "Resolve a customer by phone number or account id and return an account summary. Call this first before any account-specific action.",
        "input_schema": {
            "type": "object",
            "properties": {"phone_or_id": {"type": "string", "description": "Phone number (any format) or account id like ACC10001"}},
            "required": ["phone_or_id"],
        },
    },
    {
        "name": "get_billing",
        "description": "Get current balance, due date, and last invoice for a resolved account.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "get_data_usage",
        "description": "Get current-cycle data usage versus plan allowance for a resolved account.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "check_line_status",
        "description": "Check whether a line is active/suspended and whether there is a known outage in the account's area.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "change_plan",
        "description": "Change the account's plan. MUTATING: only call after the customer has explicitly confirmed the specific target plan. Valid plan ids: basic_5gb, plus_25gb, premium_unlimited.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}, "plan_id": {"type": "string"}},
            "required": ["account_id", "plan_id"],
        },
    },
    {
        "name": "escalate",
        "description": "Hand the conversation to a human agent. Use for billing disputes, repeated failures, legal/churn threats, or anything out of policy. Ends your control of the call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "summary": {"type": "string", "description": "Short summary of the conversation so far for the human."},
            },
            "required": ["reason"],
        },
    },
]


def get_tool(name: str) -> ToolFn | None:
    return _REGISTRY.get(name)
