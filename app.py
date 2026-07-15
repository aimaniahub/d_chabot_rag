"""
PDF RAG CLI / entrypoint

  python app.py ingest
  python app.py chat -q "..."
  python app.py status
  python app.py eval --suite predeploy
  python app.py serve
  python app.py ui          # optional local Streamlit (dev)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import API_HOST, API_PORT, UI_PORT, ensure_dirs


def cmd_ingest(args: argparse.Namespace) -> int:
    from modules.ingest import ingest_paths

    paths = [Path(args.file)] if args.file else None
    report = ingest_paths(paths=paths, rebuild=args.rebuild, force=args.force)
    print(json.dumps(report.to_dict(), indent=2))
    if report.failed and not report.indexed and not report.skipped:
        return 1
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from modules.chat import ChatService
    from modules.ingest import get_index_stats

    stats = get_index_stats()
    if stats["chunk_count"] == 0:
        print(
            "Index is empty. Add PDFs to data/ then run:\n  python app.py ingest",
            file=sys.stderr,
        )
        return 1

    service = ChatService(top_k=args.top_k, min_score=args.min_score)

    def answer_one(question: str) -> None:
        resp = service.ask(question, doc_id=args.doc_id)
        print("\nAnswer:\n")
        print(resp.answer)
        if resp.sources:
            print("\nSources:")
            for s in resp.sources:
                print(
                    f"  [{s['n']}] {s['source']} "
                    f"p.{s['page_start']}-{s['page_end']} score={s['score']}"
                )
        m = resp.metrics
        print(
            f"\nMetrics: total={m.timings.total_ms:.0f}ms "
            f"embed={m.timings.embed_ms:.0f}ms "
            f"retrieve={m.timings.retrieve_ms:.0f}ms "
            f"llm={m.timings.llm_ms:.0f}ms "
            f"tokens={m.tokens.total_tokens} "
            f"hits={m.hits_above_threshold}/{m.hits_returned}"
        )

    if args.question:
        answer_one(args.question.strip())
        return 0

    print("Interactive chat (empty line or Ctrl+C to exit). Index ready.")
    while True:
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            answer_one(q)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from modules.ingest import get_index_stats

    print(json.dumps(get_index_stats(), indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.run_eval import run_eval_suite

    report = run_eval_suite(suite=args.suite)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("passed") else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host or API_HOST
    port = args.port or API_PORT
    print(f"Starting API on http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=args.reload)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    port = args.port or UI_PORT
    ui_path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    if not ui_path.exists():
        print("Streamlit UI not found. Install dev deps: pip install streamlit", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    print(f"Starting UI on http://127.0.0.1:{port}")
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF RAG backend")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Index PDFs from data/ into Chroma")
    p_ing.add_argument("--file", help="Single PDF path")
    p_ing.add_argument("--rebuild", action="store_true")
    p_ing.add_argument("--force", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat", help="Ask questions against the index")
    p_chat.add_argument("-q", "--question")
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.add_argument("--min-score", type=float, default=None)
    p_chat.add_argument("--doc-id")
    p_chat.set_defaults(func=cmd_chat)

    p_st = sub.add_parser("status", help="Show index stats")
    p_st.set_defaults(func=cmd_status)

    p_ev = sub.add_parser("eval", help="Run evaluation suite")
    p_ev.add_argument(
        "--suite",
        choices=["smoke", "retrieval", "predeploy", "full"],
        default="predeploy",
    )
    p_ev.set_defaults(func=cmd_eval)

    p_api = sub.add_parser("serve", help="Start FastAPI server")
    p_api.add_argument("--host", default=None)
    p_api.add_argument("--port", type=int, default=None)
    p_api.add_argument("--reload", action="store_true")
    p_api.set_defaults(func=cmd_serve)

    p_ui = sub.add_parser("ui", help="Optional Streamlit UI (local/dev)")
    p_ui.add_argument("--port", type=int, default=None)
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
