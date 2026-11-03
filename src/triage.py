"""
src.triage
==========
Support ticket classification pipeline.

Stages
------
    1. Redact PII from every ticket (email, phone, order ID).
    2. Call the mock LLM; validate structured JSON output.
    3. Retry on TransientLLMError with exponential back-off (max 3 retries).
    4. Assign priority and routing based on category and customer segment.
    5. Optionally enforce a token budget — stop early when exhausted.

Public API
----------
    run_pipeline(tickets, segments, system_prompt, token_budget)  → dict
    run(root)                                                      → None

Usage
-----
    from src.triage import run
    run()

    python -m src.triage
"""

import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import mock_llm


# ── Paths ──────────────────────────────────────────────────────────────────────
def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _paths(root: Path | None = None):
    r = root or _root()
    return (
        r / "data" / "raw" / "tickets.csv",
        r / "outputs" / "customer_segments.csv",
        r / "outputs",
    )


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 400
BASE_DELAY       = 0.05
MAX_RETRIES      = 3

ALLOWED     = {"billing", "delivery", "technical", "fraud"}
CLASSIFIED  = {"classified_json", "classified_keyword"}
HUMAN_STATUSES = {"skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
SENSITIVE_PATTERNS = (EMAIL_RE, PHONE_RE, ORDER_RE)

OUTPUT_COLUMNS = ["ticket_id","customer_id","status","category",
                  "segment_name","priority","route","tokens_used"]

# ── Prompt templates ──────────────────────────────────────────────────────────
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


# ── PII redaction ─────────────────────────────────────────────────────────────
def redact_text(text: str) -> tuple[str, dict]:
    """
    Replace PII patterns with placeholder tokens.

    Returns
    -------
    (redacted_text, counts_dict)
        counts_dict keys: 'emails', 'phones', 'order_ids'

    Raises
    ------
    AssertionError if any PII pattern survives redaction.
    """
    counts: dict[str, int] = {}
    text, counts["emails"]    = EMAIL_RE.subn("[EMAIL]",    text)
    text, counts["phones"]    = PHONE_RE.subn("[PHONE]",    text)
    text, counts["order_ids"] = ORDER_RE.subn("[ORDER_ID]", text)
    assert not any(p.search(text) for p in SENSITIVE_PATTERNS), \
        "PII survived redaction!"
    return text, counts


# ── Response validation ───────────────────────────────────────────────────────
def validate_reply(content: str) -> tuple[str, str]:
    """
    Parse the LLM response and return (status, category).

    Status values
    -------------
    classified_json      — valid JSON with an allowed category
    classified_keyword   — plain text containing exactly one allowed keyword
    flagged_invalid_output — anything else
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words   = re.findall(r"[a-z]+", str(content).lower())
        matches = [w for w in words if w in ALLOWED]
        if len(matches) == 1:
            return "classified_keyword", matches[0]
        return "flagged_invalid_output", ""

    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED:
        return "classified_json", parsed["category"]
    return "flagged_invalid_output", ""


# ── Routing ───────────────────────────────────────────────────────────────────
def get_route(status: str, category: str, is_premium: bool) -> tuple[str, str]:
    """Return (priority, route) for a classified ticket."""
    if status in HUMAN_STATUSES:    return "NORMAL", "human_review"
    if category == "fraud":         return "HIGH",   "security_team"
    if is_premium:                  return "HIGH",   "premium_desk"
    return "NORMAL", "support_queue"


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(
    tickets:      list[dict],
    segments:     dict[str, dict],
    system_prompt: str,
    token_budget: int | None = None,
) -> dict:
    """
    Classify a list of ticket dicts.

    Parameters
    ----------
    tickets       : list of dicts with keys ticket_id, customer_id, text, _redacted, true_category
    segments      : dict mapping customer_id → segment dict (segment_name, is_premium)
    system_prompt : ZERO_SYSTEM or FEW_SYSTEM
    token_budget  : optional total-token ceiling; remaining tickets → budget_exhausted

    Returns
    -------
    dict with keys: rows, total_tokens, total_attempts, retry_count
    """
    results: list[dict] = []
    total_tokens = 0
    total_attempts = 0
    retry_count = 0

    mock_llm.reset()

    for ticket in tickets:
        tid  = ticket["ticket_id"]
        cid  = ticket["customer_id"]
        orig = ticket.get("text") or ""
        safe = ticket["_redacted"]
        seg  = segments.get(cid, {})
        seg_name = seg.get("segment_name", "")
        is_prem  = str(seg.get("is_premium", "")).strip().lower() in {"true","1","yes"}

        status   = ""
        category = ""
        ticket_tokens = 0

        if not orig.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(orig) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": safe},
            ]
            assert msgs[-1]["content"] == safe
            assert not any(p.search(msgs[-1]["content"]) for p in SENSITIVE_PATTERNS)

            retries_used = 0
            while True:
                if token_budget is not None and total_tokens >= token_budget:
                    status = "budget_exhausted"
                    category = ""
                    break

                total_attempts += 1
                try:
                    response = mock_llm.chat(msgs)
                    used = int(response.usage.total_tokens)
                    total_tokens  += used
                    ticket_tokens += used
                    status, category = validate_reply(response.choices[0].message.content)
                    break

                except mock_llm.TransientLLMError:
                    if retries_used >= MAX_RETRIES:
                        status = "flagged_invalid_output"
                        category = ""
                        break
                    wait = BASE_DELAY * (2 ** retries_used) * random.uniform(0.9, 1.15)
                    print(f"Retry {tid} attempt {retries_used} wait={wait:.3f}s")
                    time.sleep(wait)
                    retries_used += 1
                    retry_count  += 1

        priority, route = get_route(status, category, is_prem)

        results.append({
            "ticket_id":     tid,
            "customer_id":   cid,
            "status":        status,
            "category":      category,
            "segment_name":  seg_name,
            "priority":      priority,
            "route":         route,
            "tokens_used":   ticket_tokens,
            "true_category": ticket.get("true_category", ""),
        })

    return {
        "rows":           results,
        "total_tokens":   total_tokens,
        "total_attempts": total_attempts,
        "retry_count":    retry_count,
    }


# ── Metrics helper ────────────────────────────────────────────────────────────
def print_metrics(label: str, run_result: dict) -> None:
    rows       = run_result["rows"]
    classified = [r for r in rows if r["status"] in CLASSIFIED]
    correct    = sum(r["category"] == r["true_category"] for r in classified)
    accuracy   = correct / len(classified) if classified else 0
    print(f"\n── {label} ──")
    print(f"  Classified : {len(classified)}")
    print(f"  Correct    : {correct}/{len(classified)}")
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  Tokens     : {run_result['total_tokens']}")
    print(f"  Retries    : {run_result['retry_count']}")
    print(f"  Status cts : {dict(Counter(r['status'] for r in rows))}")


# ── I/O helpers ───────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _save_results(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})


# ── Entry point ───────────────────────────────────────────────────────────────
def run(root: Path | None = None) -> None:
    """
    Full triage pipeline:
      - zero-shot run (no budget)
      - few-shot run (no budget)
      - budget demonstration
      - final few-shot high-budget run → outputs/triage_results.csv
    """
    tickets_file, segments_file, out_dir = _paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets      = _load_csv(tickets_file)
    segment_rows = _load_csv(segments_file)
    segments     = {r["customer_id"]: r for r in segment_rows}

    # Pre-redact all tickets
    redaction_totals = Counter()
    for t in tickets:
        orig = t.get("text") or ""
        t["_original_text"] = orig
        t["_redacted"], counts = redact_text(orig)
        redaction_totals.update(counts)

    print(f"Input tickets : {len(tickets)}")
    print(f"Redactions    : {dict(redaction_totals)}")

    # Zero-shot
    zero = run_pipeline(tickets, segments, ZERO_SYSTEM)
    print_metrics("ZERO-SHOT", zero)

    # Few-shot
    few = run_pipeline(tickets, segments, FEW_SYSTEM)
    print_metrics("FEW-SHOT", few)

    # Budget demo
    budget_demo = run_pipeline(tickets, segments, ZERO_SYSTEM, token_budget=6000)
    exhausted   = sum(r["status"] == "budget_exhausted" for r in budget_demo["rows"])
    demo_budget = 6000
    if exhausted == 0:
        demo_budget = 5000
        budget_demo = run_pipeline(tickets, segments, ZERO_SYSTEM, token_budget=5000)
        exhausted   = sum(r["status"] == "budget_exhausted" for r in budget_demo["rows"])

    _save_results(budget_demo["rows"], out_dir / "triage_budget_demo.csv")
    print(f"\nBudget demo limit: {demo_budget}, exhausted: {exhausted} tickets")

    # Final submission
    final_run = run_pipeline(tickets, segments, FEW_SYSTEM, token_budget=100_000)
    _save_results(final_run["rows"], out_dir / "triage_results.csv")
    print(f"\nFinal few-shot tokens: {final_run['total_tokens']}")
    print("FINAL CHECKS PASSED")
    print("Saved: triage_results.csv, triage_budget_demo.csv")


if __name__ == "__main__":
    run()
