import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import mock_llm

ROOT = Path(__file__).resolve().parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = {"billing", "delivery", "technical", "fraud"}
CLASSIFIED = {"classified_json", "classified_keyword"}
HUMAN_STATUSES = {
    "skipped_empty", "flagged_long",
    "flagged_invalid_output", "budget_exhausted",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
SENSITIVE_PATTERNS = (EMAIL_RE, PHONE_RE, ORDER_RE)

ZERO_SYSTEM = (
    "CLASSIFY support tickets. Allowed categories: billing, delivery, "
    "technical, fraud. Return JSON only in exactly this form: "
    '{"category": "...", "reason": "..."}'
)

FEW_SYSTEM = ZERO_SYSTEM + """
EXAMPLES:
Fraud example: {"category": "fraud", "reason": "unauthorized transaction"}
Billing example: {"category": "billing", "reason": "incorrect invoice amount"}
Delivery example: {"category": "delivery", "reason": "late parcel"}
"""

OUTPUT_COLUMNS = [
    "ticket_id", "customer_id", "status", "category",
    "segment_name", "priority", "route", "tokens_used",
]


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def redact_text(text):
    counts = {}
    text, counts["emails"] = EMAIL_RE.subn("[EMAIL]", text)
    text, counts["phones"] = PHONE_RE.subn("[PHONE]", text)
    text, counts["order_ids"] = ORDER_RE.subn("[ORDER_ID]", text)

    assert not any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)
    return text, counts


def validate_reply(content):
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        matches = [word for word in words if word in ALLOWED]
        if len(matches) == 1:
            return "classified_keyword", matches[0]
        return "flagged_invalid_output", ""

    # Valid JSON with a missing or invalid category must not use keyword fallback.
    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED:
        return "classified_json", parsed["category"]

    return "flagged_invalid_output", ""


def get_route(status, category, is_premium):
    if status in HUMAN_STATUSES:
        return "NORMAL", "human_review"
    if category == "fraud":
        return "HIGH", "security_team"
    if is_premium:
        return "HIGH", "premium_desk"
    return "NORMAL", "support_queue"


def save_results(rows, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def run_pipeline(tickets, segments, system_prompt, token_budget=None):
    results = []
    total_tokens = 0
    total_attempts = 0
    retry_count = 0

    # Reset immediately before each full run.
    mock_llm.reset()

    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket.get("text") or ""
        safe_text = ticket["_redacted"]
        segment = segments.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = str(segment.get("is_premium", "")).strip().lower() in {
            "true", "1", "yes"
        }

        status = ""
        category = ""
        ticket_tokens = 0

        if not original_text.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_text},
            ]

            # The user message must be only the redacted ticket text.
            assert messages[-1]["content"] == safe_text
            assert not any(
                pattern.search(messages[-1]["content"])
                for pattern in SENSITIVE_PATTERNS
            )

            retries_used = 0
            while True:
                # Budget check happens before EVERY attempt, including retries.
                if token_budget is not None and total_tokens >= token_budget:
                    status = "budget_exhausted"
                    category = ""
                    break

                total_attempts += 1
                try:
                    response = mock_llm.chat(messages)
                    used = int(response.usage.total_tokens)
                    total_tokens += used
                    ticket_tokens += used
                    status, category = validate_reply(
                        response.choices[0].message.content
                    )
                    break

                except mock_llm.TransientLLMError:
                    if retries_used >= MAX_RETRIES:
                        status = "flagged_invalid_output"
                        category = ""
                        print(
                            f"Permanent model error after retries: {ticket_id}"
                        )
                        break

                    wait = BASE_DELAY * (2 ** retries_used) * random.uniform(
                        0.9, 1.15
                    )
                    print(
                        f"Retry ticket {ticket_id} retry {retries_used} "
                        f"wait={wait:.4f}s"
                    )
                    time.sleep(wait)
                    retries_used += 1
                    retry_count += 1

        priority, route = get_route(status, category, is_premium)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": ticket_tokens,
            "true_category": ticket.get("true_category", ""),
        })

    return {
        "rows": results,
        "total_tokens": total_tokens,
        "total_attempts": total_attempts,
        "retry_count": retry_count,
    }


def print_metrics(label, run):
    rows = run["rows"]
    classified = [row for row in rows if row["status"] in CLASSIFIED]
    correct = sum(
        row["category"] == row["true_category"] for row in classified
    )
    accuracy = correct / len(classified) if classified else 0
    tokens = run["total_tokens"]
    per_ticket = tokens / len(classified) if classified else 0

    print(f"\n--- {label} ---")
    print(f"Tickets classified: {len(classified)}")
    print(f"Correct classifications: {correct}/{len(classified)}")
    print(f"Accuracy over classified tickets: {accuracy:.1%}")
    print(f"Total successful-response tokens: {tokens}")
    print(f"Tokens per classified ticket: {per_ticket:.2f}")
    print(f"Model attempts, including failed attempts: {run['total_attempts']}")
    print(f"Recovered retries: {run['retry_count']}")
    print("Status counts:", dict(Counter(r["status"] for r in rows)))
    return classified, correct, accuracy


def print_confusion(classified):
    labels = ["billing", "delivery", "technical", "fraud"]
    matrix = Counter(
        (row["true_category"], row["category"]) for row in classified
    )
    print("\nZero-shot confusion table (rows=true, columns=predicted)")
    print("true\\pred  " + "  ".join(f"{label:>9}" for label in labels))
    for actual in labels:
        print(
            f"{actual:<10}"
            + "  ".join(f"{matrix[(actual, pred)]:>9}" for pred in labels)
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = load_csv(TICKETS_FILE)
    segment_rows = load_csv(SEGMENTS_FILE)
    segments = {row["customer_id"]: row for row in segment_rows}

    assert len(tickets) == 164, f"Expected 164 tickets, got {len(tickets)}"
    assert "EXAMPLES:" not in ZERO_SYSTEM
    assert "EXAMPLES:" in FEW_SYSTEM
    assert "Fraud example:" in FEW_SYSTEM
    assert "Billing example:" in FEW_SYSTEM

    redaction_totals = Counter()
    empty_count = 0
    long_count = 0

    # Redact every ticket before any model call.
    for ticket in tickets:
        original = ticket.get("text") or ""
        ticket["_original_text"] = original
        ticket["_redacted"], counts = redact_text(original)
        redaction_totals.update(counts)

        if not original.strip():
            empty_count += 1
        elif mock_llm.est_tokens(original) > MAX_INPUT_TOKENS:
            long_count += 1

        assert not any(
            pattern.search(ticket["_redacted"])
            for pattern in SENSITIVE_PATTERNS
        )

    print("Input tickets:", len(tickets))
    print("Empty tickets:", empty_count)
    print("Long tickets:", long_count)
    print("Redaction counts:", dict(redaction_totals))
    print("Total redactions:", sum(redaction_totals.values()))

    assert empty_count == 4, f"Expected 4 empty tickets, got {empty_count}"
    assert long_count == 4, f"Expected 4 long tickets, got {long_count}"
    assert redaction_totals["emails"] == 55
    assert redaction_totals["phones"] == 91
    assert redaction_totals["order_ids"] == 132
    assert sum(redaction_totals.values()) == 278

    eligible_count = len(tickets) - empty_count - long_count
    assert eligible_count == 156

    # Run both prompt variants with no token budget.
    zero = run_pipeline(tickets, segments, ZERO_SYSTEM, token_budget=None)
    zero_classified, zero_correct, zero_accuracy = print_metrics("ZERO-SHOT", zero)
    print_confusion(zero_classified)

    few = run_pipeline(tickets, segments, FEW_SYSTEM, token_budget=None)
    few_classified, few_correct, few_accuracy = print_metrics("FEW-SHOT", few)

    zero_status = Counter(row["status"] for row in zero["rows"])
    few_status = Counter(row["status"] for row in few["rows"])

    for label, counts in (("zero-shot", zero_status), ("few-shot", few_status)):
        print(f"\n{label} required output counts:")
        print(
            "classified_json =", counts["classified_json"],
            "classified_keyword =", counts["classified_keyword"],
            "flagged_invalid_output =", counts["flagged_invalid_output"],
        )
        assert counts["classified_json"] == 135, (label, counts)
        assert counts["classified_keyword"] == 12, (label, counts)
        assert counts["flagged_invalid_output"] == 9, (label, counts)
        assert counts["skipped_empty"] == 4
        assert counts["flagged_long"] == 4

    assert len(zero_classified) == 147
    assert len(few_classified) == 147
    assert round(zero_accuracy * 100, 1) == 85.7, zero_accuracy
    assert round(few_accuracy * 100, 1) == 100.0, few_accuracy

    print("\nFew-shot cost discussion:")
    print(
        f"Zero-shot accuracy was {zero_accuracy:.1%} "
        f"({zero_correct}/147); few-shot accuracy was {few_accuracy:.1%} "
        f"({few_correct}/147). Zero-shot used {zero['total_tokens']} tokens, "
        f"while few-shot used {few['total_tokens']} tokens."
    )
    print(
        "Few-shot examples are worth their additional token cost when the "
        "improvement in correct routing prevents meaningful customer or "
        "security errors. Here, zero-shot mistakes are concentrated in "
        "fraud tickets, so the accuracy gain is operationally important. "
        "The trade-off should still be monitored using token cost and "
        "classification quality together."
    )

    # Budget demonstration: try 6000 first, then 5000 if necessary.
    budget_demo = run_pipeline(
        tickets, segments, ZERO_SYSTEM, token_budget=6000
    )
    exhausted = sum(
        row["status"] == "budget_exhausted" for row in budget_demo["rows"]
    )
    demo_budget = 6000

    if exhausted == 0:
        demo_budget = 5000
        budget_demo = run_pipeline(
            tickets, segments, ZERO_SYSTEM, token_budget=demo_budget
        )
        exhausted = sum(
            row["status"] == "budget_exhausted"
            for row in budget_demo["rows"]
        )

    save_results(
        budget_demo["rows"], OUTPUT_DIR / "triage_budget_demo.csv"
    )
    print(f"\nBudget demo limit: {demo_budget}")
    print(f"Budget-exhausted tickets: {exhausted}")
    print(f"Successful-response tokens consumed: {budget_demo['total_tokens']}")
    assert exhausted > 0, "Budget demo must exhaust at least one ticket"

    # Final submission output: few-shot, high token budget.
    final_run = run_pipeline(
        tickets, segments, FEW_SYSTEM, token_budget=100000
    )
    final_rows = final_run["rows"]
    assert len(final_rows) == 164
    assert not any(row["status"] == "budget_exhausted" for row in final_rows)

    save_results(final_rows, OUTPUT_DIR / "triage_results.csv")

    # Final routing assertions.
    for row in final_rows:
        status = row["status"]
        category = row["category"]
        segment = segments.get(row["customer_id"], {})
        premium = str(segment.get("is_premium", "")).strip().lower() in {
            "true", "1", "yes"
        }
        expected_priority, expected_route = get_route(status, category, premium)
        assert row["priority"] == expected_priority
        assert row["route"] == expected_route

        if status in HUMAN_STATUSES:
            assert row["route"] == "human_review"
            assert row["priority"] == "NORMAL"
        elif category == "fraud":
            assert row["route"] == "security_team"
            assert row["priority"] == "HIGH"

    with (OUTPUT_DIR / "triage_results.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 164
    assert list(saved[0].keys()) == OUTPUT_COLUMNS

    print("\nFINAL CHECKS PASSED")
    print("Final few-shot output: outputs/triage_results.csv")
    print("Budget demonstration: outputs/triage_budget_demo.csv")
    print("Final rows:", len(saved))
    print("Final status counts:", dict(Counter(r["status"] for r in final_rows)))
    print("Final successful-response tokens:", final_run["total_tokens"])
    print("Routing assertions: passed")


if __name__ == "__main__":
    main()
