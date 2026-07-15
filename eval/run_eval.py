"""Evaluation suites: smoke, retrieval, predeploy, full."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    EVAL_REPORTS_DIR,
    GATE_HIT_AT_K,
    GATE_P95_LATENCY_MS,
    PROJECT_ROOT,
    TOP_K,
    ensure_dirs,
)
from modules.ingest import get_index_stats, ingest_paths
from modules.metrics import estimate_tokens_from_chars, percentile, tokens_per_minute
from modules.retriever import Retriever

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"


def _load_golden() -> list[dict[str, Any]]:
    if not GOLDEN_PATH.exists():
        return []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_report(report: dict[str, Any]) -> Path:
    ensure_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = EVAL_REPORTS_DIR / f"report_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    # also latest
    latest = EVAL_REPORTS_DIR / "latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def eval_unit_smoke() -> dict[str, Any]:
    """Fast offline checks (no Gemini)."""
    from modules.chunker import create_chunks_from_pages
    from modules.pdf_loader import PageText
    from modules.prompt_builder import build_prompt, format_context
    from modules.vector_store import RetrievedChunk

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    pages = [
        PageText(page=1, text="Alpha paragraph about cats.\n\nMore text here."),
        PageText(page=2, text="Beta paragraph about dogs and training."),
    ]
    chunks = create_chunks_from_pages(pages, doc_id="demo.pdf", source="demo.pdf")
    check("chunker_nonempty", len(chunks) >= 1, f"n={len(chunks)}")
    check("chunker_metadata", all(c.page_start >= 1 for c in chunks))

    fake_hits = [
        RetrievedChunk(
            chunk_id="c1",
            text="Cats are animals.",
            score=0.9,
            doc_id="demo.pdf",
            source="demo.pdf",
            page_start=1,
            page_end=1,
            chunk_index=0,
            metadata={},
        )
    ]
    ctx = format_context(fake_hits)
    prompt = build_prompt("What are cats?", fake_hits)
    check("context_has_citation", "[1]" in ctx)
    check("prompt_has_question", "What are cats?" in prompt)
    check("prompt_abstain_rule", "could not find" in prompt.lower())

    # empty context path
    prompt_empty = build_prompt("x", [])
    check("empty_context_string", "no relevant context" in prompt_empty.lower())

    passed = all(c["ok"] for c in checks)
    return {"name": "unit_smoke", "passed": passed, "checks": checks}


def eval_ingest_smoke() -> dict[str, Any]:
    data = PROJECT_ROOT / "data"
    pdfs = list(data.glob("*.pdf"))
    if not pdfs:
        return {
            "name": "ingest_smoke",
            "passed": False,
            "error": "no PDFs in data/",
        }
    report = ingest_paths()
    stats = get_index_stats()
    ok = report.failed == 0 and stats["chunk_count"] > 0
    return {
        "name": "ingest_smoke",
        "passed": ok,
        "ingest": report.to_dict(),
        "chunk_count": stats["chunk_count"],
    }


def eval_retrieval(golden: list[dict[str, Any]]) -> dict[str, Any]:
    stats = get_index_stats()
    if stats["chunk_count"] == 0:
        ingest_paths()
        stats = get_index_stats()
    if stats["chunk_count"] == 0:
        return {"name": "retrieval", "passed": False, "error": "empty index"}

    retriever = Retriever()
    hits_at_k = []
    latencies = []
    details = []

    items = golden or [
        {
            "id": "auto",
            "question": "What is this document about?",
            "expected_doc_id": None,
        }
    ]

    for item in items:
        q = item["question"]
        t0 = time.perf_counter()
        result = retriever.retrieve(q, top_k=TOP_K)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        expected = item.get("expected_doc_id")
        hit = True
        if expected:
            hit = any(h.doc_id == expected or h.source == expected for h in result.hits)
            # if filtered empty, check all_hits
            if not hit:
                hit = any(
                    h.doc_id == expected or h.source == expected for h in result.all_hits
                )
        hits_at_k.append(1.0 if hit else 0.0)
        details.append(
            {
                "id": item.get("id"),
                "question": q,
                "hit": hit,
                "latency_ms": ms,
                "max_score": result.max_score,
                "n_hits": len(result.hits),
                "top_sources": [
                    {"source": h.source, "score": h.score, "page": h.page_start}
                    for h in result.all_hits[:3]
                ],
            }
        )

    hit_rate = sum(hits_at_k) / len(hits_at_k) if hits_at_k else 0.0
    # For generic questions without expected_doc, hit_rate stays 1; gate only if any expected
    has_expected = any(i.get("expected_doc_id") for i in items)
    passed = (hit_rate >= GATE_HIT_AT_K) if has_expected else all(
        d["n_hits"] >= 0 for d in details
    )
    # Require at least some retrieval signal
    if all(d["max_score"] is None or d["max_score"] < 0 for d in details):
        passed = False

    return {
        "name": "retrieval",
        "passed": passed,
        "hit_at_k": hit_rate,
        "gate_hit_at_k": GATE_HIT_AT_K,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "details": details,
    }


def eval_latency_tokens(n_queries: int = 3) -> dict[str, Any]:
    """End-to-end chat latency + token/TPM (needs API key)."""
    from modules.chat import ChatService

    stats = get_index_stats()
    if stats["chunk_count"] == 0:
        return {"name": "latency_tokens", "passed": False, "error": "empty index"}

    golden = _load_golden()
    questions = [g["question"] for g in golden] if golden else []
    if not questions:
        questions = ["What is this document about?"]
    while len(questions) < n_queries:
        questions.append(questions[0])

    service = ChatService()
    totals = []
    token_counts = []
    errors = []
    rows = []
    wall_start = time.perf_counter()

    for q in questions[:n_queries]:
        try:
            resp = service.ask(q)
            totals.append(resp.metrics.timings.total_ms)
            tok = resp.metrics.tokens.total_tokens
            if tok is None:
                tok = estimate_tokens_from_chars(
                    resp.metrics.tokens.estimated_prompt_chars
                ) + estimate_tokens_from_chars(len(resp.answer))
            token_counts.append(int(tok))
            rows.append(
                {
                    "question": q,
                    "total_ms": resp.metrics.timings.total_ms,
                    "llm_ms": resp.metrics.timings.llm_ms,
                    "retrieve_ms": resp.metrics.timings.retrieve_ms,
                    "tokens": tok,
                    "hits": resp.metrics.hits_above_threshold,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    wall = time.perf_counter() - wall_start
    total_tokens = sum(token_counts)
    p95 = percentile(totals, 95)
    passed = len(errors) == 0 and (p95 <= GATE_P95_LATENCY_MS if totals else False)

    return {
        "name": "latency_tokens",
        "passed": passed,
        "errors": errors,
        "p50_ms": percentile(totals, 50),
        "p95_ms": p95,
        "gate_p95_ms": GATE_P95_LATENCY_MS,
        "total_tokens": total_tokens,
        "tpm": tokens_per_minute(total_tokens, wall),
        "wall_sec": wall,
        "queries": rows,
    }


def eval_corner_cases() -> dict[str, Any]:
    from modules.chat import ChatService
    from modules.prompt_builder import build_prompt

    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    # empty question
    try:
        ChatService().ask("   ")
        check("empty_question_raises", False)
    except ValueError:
        check("empty_question_raises", True)

    # empty hits prompt
    p = build_prompt("test", [])
    check("abstain_in_empty_prompt", "could not find" in p.lower())

    stats = get_index_stats()
    check("index_nonempty_for_predeploy", stats["chunk_count"] > 0, str(stats["chunk_count"]))

    return {
        "name": "corner_cases",
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
    }


def run_eval_suite(suite: str = "predeploy") -> dict[str, Any]:
    ensure_dirs()
    golden = _load_golden()
    parts: list[dict[str, Any]] = []

    if suite in ("smoke", "predeploy", "full", "retrieval"):
        if suite != "retrieval":
            parts.append(eval_unit_smoke())
        if suite in ("smoke", "predeploy", "full"):
            parts.append(eval_ingest_smoke())
        if suite in ("retrieval", "predeploy", "full"):
            parts.append(eval_retrieval(golden))
        if suite in ("predeploy", "full"):
            parts.append(eval_corner_cases())
        if suite == "full":
            parts.append(eval_latency_tokens())
        # predeploy includes lightweight latency only if API key present
        if suite == "predeploy":
            from config import GEMINI_API_KEY

            if GEMINI_API_KEY:
                parts.append(eval_latency_tokens(n_queries=2))
            else:
                parts.append(
                    {
                        "name": "latency_tokens",
                        "passed": True,
                        "skipped": True,
                        "reason": "no GOOGLE_API_KEY; skipped LLM latency",
                    }
                )

    passed = all(p.get("passed", False) for p in parts)
    report = {
        "suite": suite,
        "passed": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parts": parts,
    }
    path = _save_report(report)
    report["report_path"] = str(path)
    return report


if __name__ == "__main__":
    import sys

    suite = sys.argv[1] if len(sys.argv) > 1 else "predeploy"
    r = run_eval_suite(suite)
    print(json.dumps(r, indent=2, default=str))
    raise SystemExit(0 if r["passed"] else 1)
