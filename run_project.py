"""
run_project.py
==============
Single-entry-point runner for the Instamart Customer Intelligence &
Support Copilot project.

Usage
-----
    python run_project.py            # run the full pipeline
    python run_project.py --dry-run  # show steps without executing
    python run_project.py --from 3   # resume from step 3 onwards

Pipeline steps
--------------
    1  generate_data.py        – synthetic raw data & policy documents
    2  clean_data.py           – raw → clean CSVs
    3  validate_data.py        – integrity checks + SQLite load
    4  part_a_eda.py           – EDA + SQL cross-validation
    5  part_b.py               – segmentation + lapse-risk model
    6  part_c.py               – ticket triage pipeline
    7  part_d.py               – policy retrieval evaluation
    8  part_d_agent_run.py     – refund agent scenarios
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── ANSI colour helpers ────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def c(text, colour): return f"{colour}{text}{RESET}"

# ── Pipeline definition ────────────────────────────────────────────────────────
STEPS = [
    {
        "step":   1,
        "script": "generate_data.py",
        "label":  "Generate synthetic data",
        "desc":   "Creates raw CSVs and policy text files in data/raw/ and data/policies/",
        "outputs": ["data/raw/customers.csv", "data/raw/orders.csv",
                    "data/raw/products.csv",  "data/raw/tickets.csv",
                    "data/policies/refund_policy.txt"],
    },
    {
        "step":   2,
        "script": "clean_data.py",
        "label":  "Clean raw data",
        "desc":   "Normalises cities, ages, genders, amounts; resolves duplicates",
        "outputs": ["data/clean/customers.csv", "data/clean/orders.csv",
                    "data/clean/products.csv"],
    },
    {
        "step":   3,
        "script": "validate_data.py",
        "label":  "Validate & load SQLite",
        "desc":   "Runs integrity assertions and writes data/instamart.db",
        "outputs": ["data/instamart.db"],
    },
    {
        "step":   4,
        "script": "part_a_eda.py",
        "label":  "Part A — Exploratory Data Analysis",
        "desc":   "SQL × pandas cross-checks, EDA summary saved to outputs/",
        "outputs": ["outputs/part_a_eda.txt"],
    },
    {
        "step":   5,
        "script": "part_b.py",
        "label":  "Part B — Customer Intelligence",
        "desc":   "RFM segmentation (K-Means K=4), lapse-risk model (RF + LR)",
        "outputs": ["outputs/customer_segments.csv", "outputs/segment_profiles.csv",
                    "outputs/kmeans_metrics.csv",    "outputs/lapse_model.joblib",
                    "outputs/rf_top_features.csv",   "outputs/retention_list.csv"],
    },
    {
        "step":   6,
        "script": "part_c.py",
        "label":  "Part C — Support Ticket Triage",
        "desc":   "LLM classification, PII redaction, retries, token-budget demo",
        "outputs": ["outputs/triage_results.csv", "outputs/triage_budget_demo.csv"],
    },
    {
        "step":   7,
        "script": "part_d.py",
        "label":  "Part D — Policy Retrieval Evaluation",
        "desc":   "TF-IDF chunking + cosine similarity, overlap 10 vs 0 comparison",
        "outputs": ["outputs/policy_eval_overlap10.csv",
                    "outputs/policy_eval_overlap0.csv"],
    },
    {
        "step":   8,
        "script": "part_d_agent_run.py",
        "label":  "Part D — Refund Agent Scenarios",
        "desc":   "ReAct agent: 7 scenarios + token/timeout/crash-recovery demos",
        "outputs": ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl"],
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def hr(char="─", width=70):
    return c(char * width, DIM)

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def check_outputs(outputs: list[str], root: Path) -> tuple[int, int]:
    """Return (found, total) for expected output files."""
    found = sum(1 for p in outputs if (root / p).exists())
    return found, len(outputs)

def print_header():
    print()
    print(hr("═"))
    print(c("  🛒  Instamart Customer Intelligence & Support Copilot", BOLD + CYAN))
    print(c("       Full Pipeline Runner", DIM))
    print(hr("═"))
    print()

def print_step_header(step: dict):
    n     = step["step"]
    total = len(STEPS)
    label = step["label"]
    desc  = step["desc"]
    print(hr())
    print(f"  {c(f'Step {n}/{total}', BOLD + CYAN)}  {c(label, BOLD)}")
    print(f"  {c(desc, DIM)}")
    print(hr())

def print_step_result(step: dict, elapsed: float, returncode: int, root: Path):
    found, total = check_outputs(step["outputs"], root)
    status_str   = c("PASS", GREEN) if returncode == 0 else c("FAIL", RED)
    output_str   = (
        c(f"{found}/{total} outputs confirmed", GREEN)
        if found == total
        else c(f"{found}/{total} outputs found", YELLOW)
    )
    print(
        f"\n  {status_str}  {c(step['label'], BOLD)}  "
        f"{c(fmt_time(elapsed), DIM)}  ·  {output_str}"
    )

def run_step(step: dict, root: Path, dry_run: bool) -> bool:
    """Execute one pipeline step. Returns True on success."""
    print_step_header(step)

    script = root / step["script"]
    if not script.exists():
        print(c(f"  ERROR: script not found → {script}", RED))
        return False

    if dry_run:
        print(c("  [dry-run] would execute: ", DIM) + str(script))
        return True

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
    )
    elapsed = time.perf_counter() - start

    print_step_result(step, elapsed, result.returncode, root)

    if result.returncode != 0:
        print(c(f"\n  ✗  Step {step['step']} exited with code {result.returncode}.", RED))
        print(c("     Remaining steps will not run. Fix the error above and re-run.", RED))
        return False

    return True

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run the full Instamart CIS project pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing any scripts."
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Start execution from step N (default: 1)."
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None, metavar="N",
        help="Run only step N and then stop."
    )
    args = parser.parse_args()

    root         = Path(__file__).resolve().parent
    pipeline     = [s for s in STEPS if s["step"] >= args.from_step]
    if args.only_step is not None:
        pipeline = [s for s in STEPS if s["step"] == args.only_step]
        if not pipeline:
            print(c(f"No step {args.only_step} found.", RED))
            sys.exit(1)

    print_header()

    if args.dry_run:
        print(c("  DRY-RUN MODE — no scripts will be executed\n", YELLOW))

    if args.from_step > 1:
        print(c(f"  Resuming from step {args.from_step}\n", YELLOW))

    # ── Show planned steps ────────────────────────────────────────────────────
    print(c("  Pipeline:", BOLD))
    for s in STEPS:
        active = s in pipeline
        marker = c("►", CYAN) if active else c("·", DIM)
        label  = c(s["label"], "") if active else c(s["label"], DIM)
        print(f"    {marker}  Step {s['step']}  {label}")
    print()

    # ── Execute ───────────────────────────────────────────────────────────────
    total_start   = time.perf_counter()
    passed        = 0
    failed_step   = None

    for step in pipeline:
        ok = run_step(step, root, args.dry_run)
        if ok:
            passed += 1
        else:
            failed_step = step
            break
        print()

    total_elapsed = time.perf_counter() - total_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(hr("═"))
    print(c("  PIPELINE SUMMARY", BOLD))
    print(hr("─"))

    for s in STEPS:
        ran    = s in pipeline and (s["step"] <= (pipeline[passed - 1]["step"] if passed else 0) or
                                    (failed_step and s["step"] <= failed_step["step"]))
        if not ran:
            icon = c("  ○", DIM);   note = c("skipped", DIM)
        elif failed_step and s["step"] == failed_step["step"]:
            icon = c("  ✗", RED);   note = c("FAILED", RED)
        elif s in pipeline and s["step"] <= (pipeline[passed - 1]["step"] if passed else -1):
            found, total = check_outputs(s["outputs"], root)
            note = c(f"PASS  ({found}/{total} outputs)", GREEN)
            icon = c("  ✓", GREEN)
        else:
            icon = c("  ○", DIM);   note = c("skipped", DIM)

        print(f"    {icon}  Step {s['step']}  {s['label']:<42}  {note}")

    print(hr("─"))
    if failed_step:
        print(c(f"  ✗  Pipeline stopped at step {failed_step['step']}."
                f"  Elapsed: {fmt_time(total_elapsed)}", RED))
        sys.exit(1)
    else:
        steps_run = len(pipeline) if not args.dry_run else 0
        print(c(f"  ✓  All {steps_run} step(s) completed successfully."
                f"  Total time: {fmt_time(total_elapsed)}", GREEN))

    print(hr("═"))
    print()

    if not args.dry_run and not failed_step:
        print(c("  Next steps:", BOLD))
        print(f"    • Launch the dashboard  : {c('python -m streamlit run dashboard.py', CYAN)}")
        print(f"    • Open the notebook     : {c('jupyter notebook notebook/instamart_analysis.ipynb', CYAN)}")
        print(f"    • View screenshots      : {c('docs/screenshots/', CYAN)}")
        print()


if __name__ == "__main__":
    main()
