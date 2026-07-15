from modules.chunker import create_chunks, create_chunks_from_pages
from modules.pdf_loader import PageText


def test_create_chunks_overlap():
    text = "A" * 1000
    chunks = create_chunks(text, chunk_size=200)
    assert len(chunks) >= 4


def test_page_chunks_have_metadata():
    pages = [
        PageText(page=1, text="Hello world. " * 50),
        PageText(page=2, text="Second page content. " * 50),
    ]
    chunks = create_chunks_from_pages(pages, doc_id="x.pdf", source="x.pdf")
    assert chunks
    assert chunks[0].doc_id == "x.pdf"
    assert chunks[0].page_start >= 1
