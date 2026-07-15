from modules.prompt_builder import build_prompt, format_context
from modules.vector_store import RetrievedChunk


def _hit(text="context text", score=0.8):
    return RetrievedChunk(
        chunk_id="id",
        text=text,
        score=score,
        doc_id="d.pdf",
        source="d.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        metadata={},
    )


def test_prompt_includes_sources():
    p = build_prompt("Why?", [_hit()])
    assert "[1]" in p
    assert "Why?" in p
    assert "d.pdf" in p


def test_format_context_budget():
    hits = [_hit("x" * 500, score=1.0 - i * 0.01) for i in range(20)]
    ctx = format_context(hits, max_chars=1200)
    assert len(ctx) <= 1300
