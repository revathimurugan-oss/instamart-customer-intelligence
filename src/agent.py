"""
src.agent
=========
ReAct refund agent — tools, loop, checkpointing and safe fallbacks.

Tools
-----
    get_order(order_id)                          → dict
    search_policy(query)                         → dict
    calculate_refund(order_amount, delivery_minutes) → dict
    issue_refund(order_id, refund_inr)           → dict
    escalate_to_human(order_id, refund_inr)      → dict

Public API
----------
    run_agent(user_message, scenario_id, ...)    → dict
    resume_agent(run_id)                         → dict
    run(root)                                    → None  (all 7 scenarios + demos)

Usage
-----
    from src.agent import run
    run()

    python -m src.agent
"""

import csv
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import mock_llm
from src.policy import PolicyRetriever


# ── Paths ──────────────────────────────────────────────────────────────────────
def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _paths(root: Path | None = None):
    r = root or _root()
    return (
        r / "data" / "instamart.db",
        r / "data" / "raw" / "agent_scenarios.csv",
        r / "outputs" / "refund_ledger.jsonl",
        r / "outputs" / "agent_runs.jsonl",
        r / "state",
        r / "outputs",
    )


# ── Module-level retriever (set by run()) ─────────────────────────────────────
_ACTIVE_RETRIEVER: PolicyRetriever | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

def get_order(order_id: str, db_path: Path | None = None) -> dict:
    """Look up an order in the SQLite database."""
    if db_path is None:
        db_path = _root() / "data" / "instamart.db"
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {k: row[k] for k in ("order_id", "status", "order_amount", "delivery_minutes")}


def search_policy(query: str) -> dict:
    """Retrieve the three policy chunks most relevant to a query."""
    global _ACTIVE_RETRIEVER
    if _ACTIVE_RETRIEVER is None:
        _ACTIVE_RETRIEVER = PolicyRetriever()
    return {"results": _ACTIVE_RETRIEVER.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes) -> dict:
    """Apply the 45-minute rule to compute the eligible refund amount."""
    if order_amount is None or delivery_minutes is None:
        return {"refund_inr": 0.0}
    if float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _record_ledger(tool_name: str, order_id: str,
                   refund_inr: float, ledger_path: Path) -> dict:
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "tool_name":  tool_name,
        "order_id":   order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id: str, refund_inr: float,
                 ledger_path: Path | None = None) -> dict:
    """Record an automatically approved refund in the JSONL ledger."""
    if ledger_path is None:
        ledger_path = _root() / "outputs" / "refund_ledger.jsonl"
    return _record_ledger("issue_refund", order_id, refund_inr, ledger_path)


def escalate_to_human(order_id: str, refund_inr: float,
                      ledger_path: Path | None = None) -> dict:
    """Record a refund that requires human review in the JSONL ledger."""
    if ledger_path is None:
        ledger_path = _root() / "outputs" / "refund_ledger.jsonl"
    return _record_ledger("escalate_to_human", order_id, refund_inr, ledger_path)


# ── Tool spec (OpenAI-compatible) ─────────────────────────────────────────────
TOOLS_SPEC = [
    {"name": "get_order",
     "description": "Look up an order and return its status, amount and delivery time.",
     "parameters": {"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}},
    {"name": "search_policy",
     "description": "Retrieve the most relevant policy document chunks for a query.",
     "parameters": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},
    {"name": "calculate_refund",
     "description": "Calculate the eligible refund amount from order amount and delivery time.",
     "parameters": {"type":"object","properties":
         {"order_amount":{"type":"number"},"delivery_minutes":{"type":["number","null"]}},
         "required":["order_amount","delivery_minutes"]}},
    {"name": "issue_refund",
     "description": "Append an automatically approved refund to the refund ledger.",
     "parameters": {"type":"object","properties":
         {"order_id":{"type":"string"},"refund_inr":{"type":"number"}},
         "required":["order_id","refund_inr"]}},
    {"name": "escalate_to_human",
     "description": "Record a refund request for human review.",
     "parameters": {"type":"object","properties":
         {"order_id":{"type":"string"},"refund_inr":{"type":"number"}},
         "required":["order_id","refund_inr"]}},
]

TOOL_MAP = {
    "get_order":         get_order,
    "search_policy":     search_policy,
    "calculate_refund":  calculate_refund,
    "issue_refund":      issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ══════════════════════════════════════════════════════════════════════════════
# Agent internals
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedCrash(RuntimeError):
    """Used to test checkpoint recovery without losing saved progress."""


def _fallback(status: str) -> str:
    msgs = {
        "FALLBACK_MAX_ITERATIONS":
            "FALLBACK_MAX_ITERATIONS: Step limit reached. Human review required.",
        "FALLBACK_TOKEN_LIMIT":
            "FALLBACK_TOKEN_LIMIT: Token budget reached. Human review required.",
        "FALLBACK_TIMEOUT":
            "FALLBACK_TIMEOUT: Time limit reached. Human review required.",
    }
    return msgs.get(status, "FALLBACK: Human review required.")


def _agent_status(text: str) -> str:
    return text.split(":", 1)[0].strip() if ":" in text else "FALLBACK"


def _save_checkpoint(state: dict, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{state['run_id']}.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _run_agent_state(
    state:        dict,
    max_iter:     int   = 6,
    tok_limit:    int   = 6000,
    time_limit:   float = 10.0,
    crash_after:  int | None = None,
    state_dir:    Path | None = None,
    ledger_path:  Path | None = None,
) -> dict:
    """Core agent loop — shared by run_agent() and resume_agent()."""
    if state_dir is None:
        state_dir = _root() / "state"
    if ledger_path is None:
        ledger_path = _root() / "outputs" / "refund_ledger.jsonl"

    # Patch ledger path into tool closures for this run
    def _issue_refund(order_id, refund_inr):
        return issue_refund(order_id, refund_inr, ledger_path)

    def _escalate_to_human(order_id, refund_inr):
        return escalate_to_human(order_id, refund_inr, ledger_path)

    local_map = {**TOOL_MAP,
                 "issue_refund": _issue_refund,
                 "escalate_to_human": _escalate_to_human}

    state.update({"max_iterations": max_iter,
                  "token_limit": tok_limit,
                  "time_limit_seconds": time_limit})
    started = time.monotonic()

    while state["llm_calls"] < max_iter:
        if time.monotonic() - started >= time_limit:
            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = _fallback(state["status"])
            return state

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"]    += 1
        state["tokens_used"]  += getattr(response.usage, "total_tokens", 0)

        if state["tokens_used"] > tok_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = _fallback(state["status"])
            return state

        msg = response.choices[0].message
        tcs = getattr(msg, "tool_calls", None)

        if not tcs:
            state["final_answer"] = msg.content or "Human review required."
            state["status"] = _agent_status(state["final_answer"])
            return state

        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tcs]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})

        for call in calls:
            name = call["name"]
            args = call["arguments"]
            try:
                obs = local_map[name](**args)
            except KeyError:
                obs = {"error": f"Unknown tool: {name}"}
            except Exception as e:
                obs = {"error": str(e)}

            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"],
                "name": name, "content": json.dumps(obs, ensure_ascii=False),
            })
            state["steps"].append({"tool_name": name, "arguments": args, "observation": obs})
            state["tools_executed"].append(name)
            state["step_number"] += 1

            _save_checkpoint(state, state_dir)

            if crash_after is not None and len(state["tools_executed"]) >= crash_after:
                raise SimulatedCrash(
                    f"Simulated crash after {len(state['tools_executed'])} tools."
                )

    state["status"] = "FALLBACK_MAX_ITERATIONS"
    state["final_answer"] = _fallback(state["status"])
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(
    user_message:  str,
    scenario_id:   str,
    max_iter:      int   = 6,
    tok_limit:     int   = 6000,
    time_limit:    float = 10.0,
    run_id:        str | None = None,
    crash_after:   int | None = None,
    state_dir:     Path | None = None,
    ledger_path:   Path | None = None,
) -> dict:
    """
    Run the refund agent for a single customer message.

    Safety limits
    -------------
    max_iter   : max LLM calls per run (default 6)
    tok_limit  : token budget (default 6 000)
    time_limit : wall-clock seconds (default 10)

    Returns
    -------
    dict: scenario_id, status, final_answer, llm_calls, tokens_used,
          steps, tools_executed, run_id
    """
    safe_id = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in (run_id or scenario_id)
    )
    system_prompt = (
        "You are an AGENT for customer support. Follow the refund policy. "
        "Check order status before considering a refund. Only completed orders "
        "can be refunded. Delivery must exceed 45 minutes for a full refund. "
        "Never automatically issue a refund above INR 1000; escalate it to a "
        "human support lead. Use the supplied tools to verify facts."
    )
    state = {
        "run_id":         safe_id,
        "scenario_id":    scenario_id,
        "messages":       [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "tokens_used":    0,
        "step_number":    0,
        "tools_executed": [],
        "steps":          [],
        "llm_calls":      0,
    }
    result = _run_agent_state(state, max_iter, tok_limit, time_limit,
                              crash_after, state_dir, ledger_path)
    return {k: result.get(k) for k in
            ("scenario_id","status","final_answer","llm_calls",
             "tokens_used","steps","tools_executed","run_id")}


def resume_agent(run_id: str, state_dir: Path | None = None,
                 ledger_path: Path | None = None) -> dict:
    """Resume an agent run from its last saved checkpoint."""
    if state_dir is None:
        state_dir = _root() / "state"
    state = json.loads((state_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    return _run_agent_state(
        state,
        state.get("max_iterations", 6),
        state.get("token_limit", 6000),
        state.get("time_limit_seconds", 10.0),
        state_dir=state_dir,
        ledger_path=ledger_path,
    )


# ── Run log helper ─────────────────────────────────────────────────────────────
def _append_run(result: dict, runs_file: Path, scenario_id: str | None = None) -> None:
    rec = {
        "scenario_id":  scenario_id or result.get("scenario_id"),
        "status":       result.get("status", "?"),
        "final_answer": result.get("final_answer", ""),
        "llm_calls":    result.get("llm_calls", 0),
        "tokens_used":  result.get("tokens_used", 0),
        "steps":        result.get("steps", []),
    }
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    with runs_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(root: Path | None = None) -> None:
    """Run all 7 scenarios + token / timeout / crash-recovery demos."""
    global _ACTIVE_RETRIEVER
    db_path, scenarios_file, ledger_path, runs_file, state_dir, out_dir = _paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    _ACTIVE_RETRIEVER = PolicyRetriever(
        policy_dir=(root or _root()) / "data" / "policies"
    )

    # Reset log files
    for p in (runs_file, ledger_path):
        p.write_text("", encoding="utf-8")

    with scenarios_file.open("r", encoding="utf-8", newline="") as f:
        scenarios = list(csv.DictReader(f))
    sc_lookup = {r["scenario_id"]: r for r in scenarios}

    print("\nPART D — REFUND AGENT SCENARIOS")
    print("-" * 76)

    results = []
    for sc in scenarios:
        mock_llm.reset()
        r = run_agent(sc["message"], sc["scenario_id"],
                      max_iter=6, tok_limit=6000, time_limit=10,
                      state_dir=state_dir, ledger_path=ledger_path)
        _append_run(r, runs_file)
        results.append(r)

        exp = sc["expected_outcome"]
        act = "FALLBACK" if r["status"].startswith("FALLBACK") else r["status"]
        ok  = act == exp
        print(f"{sc['scenario_id']}: expected={exp:<12} actual={act:<12} "
              f"calls={r['llm_calls']} tokens={r['tokens_used']} {'PASS' if ok else 'CHECK'}")
        print(f"  → {r['final_answer']}")

    # Token-limit demo
    s1 = sc_lookup["S1"]
    mock_llm.reset()
    td = run_agent(s1["message"], "S1_TOKEN_DEMO", tok_limit=200,
                   run_id="S1_TOKEN_DEMO", state_dir=state_dir, ledger_path=ledger_path)
    _append_run(td, runs_file)
    print(f"\nToken demo : {td['status']} (tokens={td['tokens_used']})")

    # Timeout demo
    mock_llm.reset()
    tt = run_agent(s1["message"], "S1_TIMEOUT_DEMO", time_limit=0,
                   run_id="S1_TIMEOUT_DEMO", state_dir=state_dir, ledger_path=ledger_path)
    _append_run(tt, runs_file)
    print(f"Timeout demo: {tt['status']} (calls={tt['llm_calls']})")

    # Crash-recovery demo
    crash_id   = "S1_crash_resume"
    crash_path = state_dir / f"{crash_id}.json"
    if crash_path.exists():
        crash_path.unlink()

    mock_llm.reset()
    try:
        run_agent(s1["message"], "S1_CRASH_RESUME", run_id=crash_id,
                  crash_after=2, state_dir=state_dir, ledger_path=ledger_path)
    except SimulatedCrash as e:
        print(f"\nCrash simulated: {e}")

    crashed = json.loads(crash_path.read_text(encoding="utf-8"))
    print("Tools before crash:", crashed["tools_executed"])

    crash_rec = {
        "scenario_id": "S1_CRASH_PARTIAL",
        "status": "SIMULATED_CRASH",
        "final_answer": "Stopped after saving the second tool observation.",
        "llm_calls": crashed.get("llm_calls", 0),
        "tokens_used": crashed.get("tokens_used", 0),
        "steps": crashed.get("steps", []),
    }
    _append_run(crash_rec, runs_file, scenario_id="S1_CRASH_PARTIAL")

    mock_llm.reset()
    resumed = resume_agent(crash_id, state_dir=state_dir, ledger_path=ledger_path)
    _append_run(resumed, runs_file, scenario_id="S1_CRASH_RESUME")
    print(f"Resumed status : {resumed.get('status')}")
    print(f"Final answer   : {resumed.get('final_answer')}")
    print(f"Tools used     : {resumed.get('tools_executed')}")

    expected_tools = ["get_order","search_policy","calculate_refund","issue_refund"]
    ok = (resumed.get("status") == "REFUND"
          and resumed.get("tools_executed") == expected_tools)
    print(f"Crash-recovery : {'PASS' if ok else 'CHECK'}")

    all_pass = all(
        ("FALLBACK" if r["status"].startswith("FALLBACK") else r["status"])
        == sc["expected_outcome"]
        for r, sc in zip(results, scenarios)
    )
    print(f"\nAll scenario checks: {'PASS' if all_pass else 'CHECK FAILED'}")
    print(f"Run log    → {runs_file}")
    print(f"Ledger     → {ledger_path}")
    print(f"Checkpoints→ {state_dir}")


if __name__ == "__main__":
    run()
