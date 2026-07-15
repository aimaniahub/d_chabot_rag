# PDF RAG Backend

Production-style **PDF RAG API**: ingest PDFs → Chroma vectors → grounded Gemini answers with citations.

Designed as a **reusable backend** for a separate frontend (Next.js/Vercel, React, etc.).

| Layer | Choice |
|-------|--------|
| Embeddings | **fastembed** (ONNX) — no PyTorch / torchvision |
| Vector store | **Chroma** (persistent) |
| LLM | **Gemini** via `google-genai` |
| API | **FastAPI** |
| Deploy | **Docker** → Railway / any container host |

Design notes: [`BLUEPRINT.md`](./BLUEPRINT.md) · Deploy guide: [`docs/DEPLOY.md`](./docs/DEPLOY.md)

---

## Quick start (local)

```bash
conda activate pdf_rag
pip install -r requirements.txt
# optional local UI/tests:
# pip install -r requirements-dev.txt

copy .env.example .env   # set GOOGLE_API_KEY

# Put PDFs in data/
python app.py ingest --rebuild
python app.py chat -q "What is Core Technologies?"
python app.py status
python app.py eval --suite predeploy

# Backend API (what your frontend will call)
python app.py serve
# → http://127.0.0.1:8000/docs
```

---

## Docker (recommended for servers)

```bash
# .env must contain GOOGLE_API_KEY
docker compose up --build
```

- API: http://localhost:8000  
- Health: http://localhost:8000/health  
- OpenAPI: http://localhost:8000/docs  

Persist volumes: `./data`, `./storage`.

Railway uses the same `Dockerfile` + `railway.toml`. Full steps: **[docs/DEPLOY.md](./docs/DEPLOY.md)**.

> **Vercel** = frontend only. This Python RAG service needs Docker (Railway, etc.).

---

## Connect a separate frontend

```ts
const API = process.env.NEXT_PUBLIC_RAG_API_URL;

// Chat
const r = await fetch(`${API}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What are the core services?" }),
});
const { answer, sources, metrics } = await r.json();

// Upload PDF
const form = new FormData();
form.append("file", file);
await fetch(`${API}/ingest/upload`, { method: "POST", body: form });
```

Set backend CORS to your Vercel origin:

```env
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

---

## CLI

| Command | Purpose |
|---------|---------|
| `python app.py ingest` | Index `data/*.pdf` (skip unchanged) |
| `python app.py ingest --rebuild` | Wipe + reindex |
| `python app.py chat -q "..."` | Query index |
| `python app.py status` | Chunk / doc counts |
| `python app.py eval --suite predeploy` | Latency, Hit@k, TPM gates |
| `python app.py serve` | FastAPI server |
| `python app.py ui` | Optional Streamlit (dev deps) |

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Index ready |
| GET | `/stats` | Documents + chunks |
| POST | `/ingest` | Index PDFs |
| POST | `/ingest/upload` | Upload + index PDF |
| POST | `/chat` | Question → answer + sources + metrics |

---

## Project layout

```text
pdf_rag_v2_starter/
├── api/main.py          # FastAPI (backend entry for Docker)
├── app.py               # CLI
├── config.py            # env config
├── modules/             # ingest, embed, retrieve, chat, llm, chroma
├── eval/                # golden set + predeploy suite
├── tests/               # unit tests
├── ui/                  # optional Streamlit (local only)
├── data/                # source PDFs
├── storage/             # Chroma + manifest (gitignored)
├── Dockerfile
├── docker-compose.yml
├── railway.toml
├── requirements.txt     # production
├── requirements-dev.txt # pytest + streamlit
└── docs/DEPLOY.md
```

---

## Why no torch / torchvision?

Older stack used `sentence-transformers` + PyTorch. On many machines Torch installs **without** torchvision and prints noisy import errors.

**fastembed** embeds with ONNX Runtime — smaller Docker images, no torch stack, same MiniLM model id:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

After changing embedding model, always:

```bash
python app.py ingest --rebuild
```

---

## Env vars (main)

| Var | Default | Notes |
|-----|---------|--------|
| `GOOGLE_API_KEY` | — | Required |
| `CORS_ORIGINS` | `*` | Comma-separated frontend URLs in prod |
| `PORT` | `8000` | Railway injects this |
| `AUTO_INGEST_ON_START` | `false` | `true` in Docker compose default |
| `TOP_K` / `MIN_SCORE` | `5` / `0.30` | Retrieval |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `100` | Chunking |

See `.env.example`.

---

## Eval

```bash
python app.py eval --suite predeploy
# writes eval_reports/latest.json
```

Measures retrieval Hit@k, latency p50/p95, tokens / TPM, corner cases.
