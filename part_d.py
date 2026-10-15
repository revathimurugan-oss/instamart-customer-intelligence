
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy "
    "context. If the supplied context does not contain the answer, respond "
    "with a sentence beginning exactly: I could not find. Do not invent "
    "policy facts."
)


def chunk_documents(overlap=DEFAULT_OVERLAP):
    """Split each policy document into overlapping word chunks."""
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("overlap must be between 0 and 39")

    chunks = []
    step = CHUNK_SIZE - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0

        while start < len(words):
            selected = words[start:start + CHUNK_SIZE]
            if not selected:
                break

            chunks.append({
                "source": path.name,
                "text": " ".join(selected),
            })

            if start + CHUNK_SIZE >= len(words):
                break

            start += step

    return chunks


class PolicyRetriever:
    def __init__(self, overlap=DEFAULT_OVERLAP):
        self.chunks = chunk_documents(overlap)
        if not self.chunks:
            raise ValueError(f"No policy chunks found in {POLICY_DIR}")

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        # Stable ordering makes ties reproducible.
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: (-scores[i], i),
        )

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked[:min(k, len(ranked))]
        ]


def answer(question, retriever=None):
    """Return the mock model's answer and the retrieved source filenames."""
    import mock_llm

    if retriever is None:
        retriever = PolicyRetriever()

    results = retriever.retrieve(question, k=3)

    context_parts = []
    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    user_message = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + question
    )

    response = mock_llm.chat([
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": user_message},
    ])

    return {
        "answer": response.choices[0].message.content,
        "sources": [result["source"] for result in results],
        "retrieval": results,
        "tokens_used": response.usage.total_tokens,
    }


def evaluate(retriever, evaluation_rows):
    """Evaluate retrieval, exact expected-fact substrings, and fallback."""
    answerable = [r for r in evaluation_rows if r["expected_doc"] != "NONE"]
    unanswerable = [r for r in evaluation_rows if r["expected_doc"] == "NONE"]

    retrieval_hits = 0
    fact_hits = 0
    fallback_hits = 0
    details = []

    for row in evaluation_rows:
        result = answer(row["question"], retriever)
        text = result["answer"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        if expected_doc != "NONE":
            retrieved_ok = expected_doc in result["sources"]
            fact_ok = bool(expected_fact) and expected_fact in text
            retrieval_hits += int(retrieved_ok)
            fact_hits += int(fact_ok)
        else:
            retrieved_ok = False
            fact_ok = False
            fallback_hits += int(text.startswith("I could not find"))

        details.append({
            "question": row["question"],
            "expected_doc": expected_doc,
            "sources": ";".join(result["sources"]),
            "answer": text,
            "retrieval_hit": retrieved_ok,
            "expected_fact": expected_fact,
            "fact_hit": fact_ok,
            "fallback_hit": (
                text.startswith("I could not find")
                if expected_doc == "NONE" else ""
            ),
        })

    return {
        "answerable_count": len(answerable),
        "retrieval_hits": retrieval_hits,
        "fact_hits": fact_hits,
        "unanswerable_count": len(unanswerable),
        "fallback_hits": fallback_hits,
        "details": details,
    }


def save_evaluation(details, filename):
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)


def print_evaluation(label, report):
    print(f"\n--- {label} ---")
    print(
        f"Expected document in top 3: "
        f"{report['retrieval_hits']}/{report['answerable_count']}"
    )
    print(
        f"Expected fact in answer: "
        f"{report['fact_hits']}/{report['answerable_count']}"
    )
    print(
        f"Unanswerable fallback: "
        f"{report['fallback_hits']}/{report['unanswerable_count']}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with EVAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        evaluation_rows = list(csv.DictReader(file))

    print("Policy documents:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Evaluation questions:", len(evaluation_rows))

    overlapping = PolicyRetriever(overlap=10)
    print("\nOverlap 10 chunk counts:")
    counts = {}
    for chunk in overlapping.chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print("Total chunks:", len(overlapping.chunks))

    assert len(overlapping.chunks) == 15, (
        f"Expected 15 chunks, got {len(overlapping.chunks)}. "
        f"Counts: {counts}"
    )

    top_three = overlapping.retrieve("refund policy late delivery", k=3)
    print("\nTop 3 for 'refund policy late delivery':")
    for item in top_three:
        print(f"  {item['source']} score={item['score']:.4f}")
    assert all(
        item["source"] == "refund_policy.txt" for item in top_three
    ), "Expected all three top chunks to come from refund_policy.txt"

    overlap_report = evaluate(overlapping, evaluation_rows)
    print_evaluation("OVERLAP 10", overlap_report)
    save_evaluation(
        overlap_report["details"], "policy_eval_overlap10.csv"
    )

    no_overlap = PolicyRetriever(overlap=0)
    no_overlap_report = evaluate(no_overlap, evaluation_rows)
    print_evaluation("OVERLAP 0", no_overlap_report)
    save_evaluation(
        no_overlap_report["details"], "policy_eval_overlap0.csv"
    )

    broken = [
        row for row in no_overlap_report["details"]
        if row["expected_doc"] != "NONE" and not row["fact_hit"]
    ]
    print("\nQuestions failing expected-fact check with zero overlap:")
    for row in broken:
        print(f"  - {row['question']}")
        print(f"    Expected fact: {row['expected_fact']}")
        print(f"    Answer: {row['answer']}")

    print("\nTF-IDF explanation:")
    print(
        "Term frequency (TF) measures how strongly a term appears within "
        "a chunk, while inverse document frequency (IDF) gives less weight "
        "to terms common across many chunks. TF-IDF is lexical rather than "
        "semantic, so two phrasings with the same meaning but different "
        "words may have low similarity."
    )

    print("\nSaved evaluation CSV files.")
    print("This step tests retrieval and grounding only; refund-agent tools "
          "will be added next.")


if __name__ == "__main__":
    main()

# ============================================================
# PART D — REFUND AGENT, TOOLS, CHECKPOINTS AND SAFE FALLBACKS
# ============================================================

import json
import sqlite3
import time
from datetime import datetime, timezone

import mock_llm

DB_FILE = ROOT / "data" / "instamart.db"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
STATE_DIR = ROOT / "state"

ACTIVE_RETRIEVER = None


def get_order(order_id):
    """Look up an order in the Part A SQLite database."""
    with sqlite3.connect(DB_FILE) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": row["order_amount"],
        "delivery_minutes": row["delivery_minutes"],
    }


def search_policy(query):
    """Retrieve the three policy chunks most relevant to a query."""
    results = ACTIVE_RETRIEVER.retrieve(query, k=3)
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate a refund only when delivery exceeded 45 minutes."""
    if order_amount is None or delivery_minutes is None:
        return {"refund_inr": 0.0}

    if float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}

    return {"refund_inr": 0.0}


def _record_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    """Record a refund that has been approved for automatic issuance."""
    return _record_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    """Record a refund request that requires human approval."""
    return _record_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order and return its status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the most relevant policy document chunks for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund amount from order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Append an automatically approved refund to the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request for human review rather than issuing it automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

class SimulatedCrash(RuntimeError):
    """Used to test checkpoint recovery without losing saved progress."""


def _agent_status(final_answer):
    """Extract the status prefix, such as REFUND or DENIED."""
    if ":" in final_answer:
        return final_answer.split(":", 1)[0].strip()
    return "FALLBACK"



def _fallback(status):
    messages = {
        "FALLBACK_MAX_ITERATIONS":
            "FALLBACK_MAX_ITERATIONS: A human support agent must review this request because the step limit was reached.",
        "FALLBACK_TOKEN_LIMIT":
            "FALLBACK_TOKEN_LIMIT: A human support agent must review this request because the token budget was reached.",
        "FALLBACK_TIMEOUT":
            "FALLBACK_TIMEOUT: A human support agent must review this request because the time limit was reached.",
    }
    return messages.get(
        status,
        "FALLBACK: A human support agent must review this request.",
    )



def _save_checkpoint(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{state['run_id']}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)


def _run_agent_state(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    started = time.monotonic()
    state["max_iterations"] = max_iterations
    state["token_limit"] = token_limit
    state["time_limit_seconds"] = time_limit_seconds

    while state["llm_calls"] < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = _fallback(state["status"])
            return state

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1

        usage = getattr(response, "usage", None)
        state["tokens_used"] += getattr(usage, "total_tokens", 0)

        if state["tokens_used"] > token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = _fallback(state["status"])
            return state

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            state["final_answer"] = message.content or (
                "A human support agent must review this request."
            )
            state["status"] = _agent_status(state["final_answer"])
            return state

        assistant_calls = []
        for call in tool_calls:
            assistant_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        for call in assistant_calls:
            tool_name = call["name"]
            arguments = call["arguments"]

            if tool_name not in TOOL_MAP:
                observation = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as error:
                    observation = {"error": str(error)}

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": tool_name,
                "content": json.dumps(observation, ensure_ascii=False),
            })

            state["steps"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            state["tools_executed"].append(tool_name)
            state["step_number"] += 1

            # Save progress immediately after every tool observation.
            _save_checkpoint(state)

            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) >= crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {len(state['tools_executed'])} tools."
                )

    state["status"] = "FALLBACK_MAX_ITERATIONS"
    state["final_answer"] = _fallback(state["status"])
    return state


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    """Run the refund agent with tool, token, time and step limits."""
    safe_id = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in (run_id or scenario_id)
    )

    system_prompt = (
        "You are an AGENT for customer support. Follow the refund policy. "
        "Check order status before considering a refund. Only completed orders "
        "can be refunded. Delivery must exceed 45 minutes for a full refund. "
        "Never automatically issue a refund above INR 1000; escalate it to a "
        "human support lead. Use the supplied tools to verify facts."
    )

    state = {
        "run_id": safe_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
    }

    result = _run_agent_state(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools,
    )
    return {
        "scenario_id": result["scenario_id"],
        "status": result.get("status", _agent_status(result.get("final_answer", ""))),
        "final_answer": result.get("final_answer", ""),
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
        "tools_executed": result["tools_executed"],
        "run_id": result["run_id"],
    }


def resume_agent(run_id):
    """Resume from the last saved checkpoint without repeating completed tools."""
    path = STATE_DIR / f"{run_id}.json"
    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    return _run_agent_state(
        state,
        max_iterations=state.get("max_iterations", 6),
        token_limit=state.get("token_limit", 6000),
        time_limit_seconds=state.get("time_limit_seconds", 10),
    )

def _append_agent_run(result, scenario_id=None):
    """Append one run record to the JSONL log."""
    record = {
        "scenario_id": scenario_id or result.get("scenario_id"),
        "status": result.get("status", "SIMULATED_CRASH"),
        "final_answer": result.get("final_answer", ""),
        "llm_calls": result.get("llm_calls", 0),
        "tokens_used": result.get("tokens_used", 0),
        "steps": result.get("steps", []),
    }

    AGENT_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_RUNS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _public_agent_result(state, scenario_id=None):
    """Convert a checkpoint or resumed state into the standard result format."""
    return {
        "scenario_id": scenario_id or state.get("scenario_id"),
        "status": state.get("status", "FALLBACK"),
        "final_answer": state.get("final_answer", ""),
        "llm_calls": state.get("llm_calls", 0),
        "tokens_used": state.get("tokens_used", 0),
        "steps": state.get("steps", []),
        "tools_executed": state.get("tools_executed", []),
        "run_id": state.get("run_id", ""),
    }

def run_agent_main():
    """Run all required Part D scenarios and safety demonstrations."""
    global ACTIVE_RETRIEVER

    ACTIVE_RETRIEVER = PolicyRetriever()

    for path in (AGENT_RUNS_FILE, LEDGER_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    with SCENARIOS_FILE.open("r", encoding="utf-8", newline="") as file:
        scenarios = list(csv.DictReader(file))

    scenario_lookup = {row["scenario_id"]: row for row in scenarios}
    results = []

    print("\nPART D — REFUND AGENT SCENARIOS")
    print("-" * 76)

    for scenario in scenarios:
        mock_llm.reset()
        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario["scenario_id"],
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
        )

        _append_agent_run(result)
        results.append(result)

        expected = scenario["expected_outcome"]
        actual = (
            "FALLBACK"
            if result["status"].startswith("FALLBACK")
            else result["status"]
        )
        passed = actual == expected

        print(
            f"{scenario['scenario_id']}: "
            f"expected={expected}, actual={actual}, "
            f"calls={result['llm_calls']}, "
            f"tokens={result['tokens_used']}, "
            f"{'PASS' if passed else 'CHECK'}"
        )
        print(f"  Answer: {result['final_answer']}")

    # Token-budget demonstration.
    s1 = scenario_lookup["S1"]
    mock_llm.reset()
    token_result = run_agent(
        s1["message"],
        "S1_TOKEN_DEMO",
        token_limit=200,
        run_id="S1_TOKEN_DEMO",
    )
    _append_agent_run(token_result)
    print(
        "\nToken-limit demo:",
        token_result["status"],
        f"(tokens={token_result['tokens_used']})",
    )

    # Timeout demonstration.
    mock_llm.reset()
    timeout_result = run_agent(
        s1["message"],
        "S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
        run_id="S1_TIMEOUT_DEMO",
    )
    _append_agent_run(timeout_result)
    print(
        "Timeout demo:",
        timeout_result["status"],
        f"(calls={timeout_result['llm_calls']})",
    )

    # Crash after the second tool observation has been checkpointed.
    crash_id = "S1_crash_resume"
    checkpoint_path = STATE_DIR / f"{crash_id}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    mock_llm.reset()
    try:
        run_agent(
            s1["message"],
            "S1_CRASH_RESUME",
            run_id=crash_id,
            crash_after_tools=2,
        )
    except SimulatedCrash as error:
        print("\nCrash simulation:", str(error))

    with checkpoint_path.open("r", encoding="utf-8") as file:
        crashed_state = json.load(file)

    print("Tools completed before crash:", crashed_state["tools_executed"])

    crash_record = _public_agent_result(
        crashed_state, scenario_id="S1_CRASH_PARTIAL"
    )
    crash_record["status"] = "SIMULATED_CRASH"
    crash_record["final_answer"] = (
        "Stopped after saving the second tool observation."
    )
    _append_agent_run(crash_record)

    mock_llm.reset()
    resumed_state = resume_agent(crash_id)
    resumed_result = _public_agent_result(
        resumed_state, scenario_id="S1_CRASH_RESUME"
    )
    _append_agent_run(resumed_result)

    print("Resumed status:", resumed_result["status"])
    print("Final answer:", resumed_result["final_answer"])
    print("Tools across crash + resume:", resumed_result["tools_executed"])

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]
    crash_passed = (
        resumed_result["status"] == "REFUND"
        and resumed_result["tools_executed"] == expected_tools
    )
    print("Crash-recovery check:", "PASS" if crash_passed else "CHECK")

    expected_all = all(
        (
            "FALLBACK"
            if result["status"].startswith("FALLBACK")
            else result["status"]
        ) == scenario["expected_outcome"]
        for result, scenario in zip(results, scenarios)
    )

    print("\nRun log saved to:", AGENT_RUNS_FILE)
    print("Refund/escalation ledger saved to:", LEDGER_FILE)
    print("Checkpoint files saved in:", STATE_DIR)
    print(
        "\nScenario outcome check:",
        "PASS" if expected_all else "CHECK FAILED",
    )


if __name__ == "__main__":
    run_agent_main()
