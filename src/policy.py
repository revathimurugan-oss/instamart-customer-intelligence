"""
src.policy
==========
TF-IDF policy retrieval and evaluation.

Classes
-------
    PolicyRetriever   — chunk documents, build TF-IDF index, retrieve top-k chunks

Functions
---------
    answer(question, retriever)           → dict
    evaluate(retriever, eval_rows)        → dict
    run(root)                             → None

Usage
-----
    from src.policy import PolicyRetriever, answer, run
    r = PolicyRetriever(overlap=10)
    result = answer("Who approves a refund above INR 1000?", r)
    print(result["answer"])

    # or run full evaluation:
    run()

    python -m src.policy
"""

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Paths ──────────────────────────────────────────────────────────────────────
def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _paths(root: Path | None = None):
    r = root or _root()
    return (
        r / "data" / "policies",
        r / "data" / "raw" / "policy_eval.csv",
        r / "outputs",
    )


# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 40
DEFAULT_OVERLAP = 10

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy "
    "context. If the supplied context does not contain the answer, respond "
    "with a sentence beginning exactly: I could not find. Do not invent "
    "policy facts."
)


# ── Document chunking ─────────────────────────────────────────────────────────
def chunk_documents(policy_dir: Path, overlap: int = DEFAULT_OVERLAP) -> list[dict]:
    """
    Split each .txt policy file into overlapping word-level chunks.

    Parameters
    ----------
    policy_dir : directory containing *.txt policy files
    overlap    : number of words shared between consecutive chunks (0 to CHUNK_SIZE-1)

    Returns
    -------
    list of dicts with keys: source (filename), text (chunk content)
    """
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("overlap must be between 0 and 39")

    chunks: list[dict] = []
    step = CHUNK_SIZE - overlap

    for path in sorted(policy_dir.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            selected = words[start: start + CHUNK_SIZE]
            if not selected:
                break
            chunks.append({"source": path.name, "text": " ".join(selected)})
            if start + CHUNK_SIZE >= len(words):
                break
            start += step

    return chunks


# ── Retriever ─────────────────────────────────────────────────────────────────
class PolicyRetriever:
    """
    TF-IDF retriever over policy document chunks.

    Parameters
    ----------
    overlap     : word overlap between consecutive chunks (default 10)
    policy_dir  : directory containing *.txt policy files
    """

    def __init__(self, overlap: int = DEFAULT_OVERLAP, policy_dir: Path | None = None):
        if policy_dir is None:
            policy_dir = _root() / "data" / "policies"

        self.chunks    = chunk_documents(policy_dir, overlap)
        if not self.chunks:
            raise ValueError(f"No policy chunks found in {policy_dir}")

        self.vectorizer = TfidfVectorizer()
        self.matrix     = self.vectorizer.fit_transform(
            [c["text"] for c in self.chunks]
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Return the top-k most relevant chunks for a query.

        Returns
        -------
        list of dicts: source, score, text — sorted by descending score,
        ties broken by original chunk index (stable / reproducible).
        """
        qv     = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).ravel()
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [
            {"source": self.chunks[i]["source"],
             "score":  float(scores[i]),
             "text":   self.chunks[i]["text"]}
            for i in ranked[: min(k, len(ranked))]
        ]


# ── Q&A function ──────────────────────────────────────────────────────────────
def answer(question: str, retriever: PolicyRetriever | None = None) -> dict:
    """
    Retrieve relevant policy chunks and return the mock LLM's grounded answer.

    Returns
    -------
    dict with keys: answer, sources (list of filenames), retrieval (full results),
    tokens_used
    """
    import mock_llm

    if retriever is None:
        retriever = PolicyRetriever()

    results = retriever.retrieve(question, k=3)
    context = "\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
    user_msg = f"CONTEXT:\n{context}\nQUESTION:\n{question}"

    response = mock_llm.chat([
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user",   "content": user_msg},
    ])

    return {
        "answer":      response.choices[0].message.content,
        "sources":     [r["source"] for r in results],
        "retrieval":   results,
        "tokens_used": response.usage.total_tokens,
    }


# ── Batch evaluation ──────────────────────────────────────────────────────────
def evaluate(retriever: PolicyRetriever, eval_rows: list[dict]) -> dict:
    """
    Evaluate retrieval quality over a set of Q&A pairs.

    Metrics
    -------
    retrieval_hit  : expected document appears in top-3 retrieved chunks
    fact_hit       : expected fact substring appears in the answer
    fallback_hit   : unanswerable question triggers "I could not find…" response

    Returns
    -------
    dict: answerable_count, retrieval_hits, fact_hits,
          unanswerable_count, fallback_hits, details (list of row dicts)
    """
    answerable   = [r for r in eval_rows if r["expected_doc"] != "NONE"]
    unanswerable = [r for r in eval_rows if r["expected_doc"] == "NONE"]

    ret_hits = 0; fact_hits = 0; fallback_hits = 0
    details: list[dict] = []

    for row in eval_rows:
        res  = answer(row["question"], retriever)
        txt  = res["answer"]
        edoc = row["expected_doc"]
        efact = row["expected_fact"]

        if edoc != "NONE":
            rh = edoc in res["sources"]
            fh = bool(efact) and efact in txt
            ret_hits  += int(rh)
            fact_hits += int(fh)
        else:
            rh = False; fh = False
            fallback_hits += int(txt.startswith("I could not find"))

        details.append({
            "question":      row["question"],
            "expected_doc":  edoc,
            "sources":       ";".join(res["sources"]),
            "answer":        txt,
            "retrieval_hit": rh,
            "expected_fact": efact,
            "fact_hit":      fh,
            "fallback_hit":  txt.startswith("I could not find") if edoc == "NONE" else "",
        })

    return {
        "answerable_count":   len(answerable),
        "retrieval_hits":     ret_hits,
        "fact_hits":          fact_hits,
        "unanswerable_count": len(unanswerable),
        "fallback_hits":      fallback_hits,
        "details":            details,
    }


# ── Save evaluation CSV ───────────────────────────────────────────────────────
def save_evaluation(details: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)


# ── Entry point ───────────────────────────────────────────────────────────────
def run(root: Path | None = None) -> None:
    """Run the full policy retrieval evaluation (overlap=10 and overlap=0)."""
    policy_dir, eval_file, out_dir = _paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    with eval_file.open("r", encoding="utf-8-sig", newline="") as f:
        eval_rows = list(csv.DictReader(f))

    print(f"Policy documents : {len(list(policy_dir.glob('*.txt')))}")
    print(f"Eval questions   : {len(eval_rows)}")

    r10 = PolicyRetriever(overlap=10, policy_dir=policy_dir)
    r0  = PolicyRetriever(overlap=0,  policy_dir=policy_dir)

    print(f"\nChunks (overlap=10) : {len(r10.chunks)}")
    print(f"Chunks (overlap=0)  : {len(r0.chunks)}")

    rep10 = evaluate(r10, eval_rows)
    rep0  = evaluate(r0,  eval_rows)

    n_ans = rep10["answerable_count"]
    print(f"\nOverlap 10 → retrieval {rep10['retrieval_hits']}/{n_ans}, "
          f"fact {rep10['fact_hits']}/{n_ans}, "
          f"fallback {rep10['fallback_hits']}/{rep10['unanswerable_count']}")
    print(f"Overlap  0 → retrieval {rep0['retrieval_hits']}/{n_ans}, "
          f"fact {rep0['fact_hits']}/{n_ans}, "
          f"fallback {rep0['fallback_hits']}/{rep0['unanswerable_count']}")

    save_evaluation(rep10["details"], out_dir / "policy_eval_overlap10.csv")
    save_evaluation(rep0["details"],  out_dir / "policy_eval_overlap0.csv")
    print("\nSaved: policy_eval_overlap10.csv, policy_eval_overlap0.csv")


if __name__ == "__main__":
    run()
