# Deploy PDF RAG Backend (Docker)

This repo is the **backend API**. Deploy it with Docker (Railway, Fly.io, ECS, VPS).  
Deploy your **frontend** separately (e.g. Vercel) and call this API.

```
[ Vercel Frontend ]  --HTTPS JSON-->  [ Docker: FastAPI + Chroma + fastembed + Gemini ]
```

## Why not Vercel for this backend?

Vercel is ideal for Next.js frontends. This service needs:

- long-running process
- local/persistent vector storage
- embedding model files
- multi-MB Docker image

Use **Railway (or any Docker host)** for the API; **Vercel** for the UI.

---

## Local Docker

```bash
# From project root
cp .env.example .env   # set GOOGLE_API_KEY

docker compose up --build
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

Volumes mount `./data` and `./storage` so the index survives restarts.

---

## Railway

1. Create a new Railway project → **Deploy from GitHub** (or CLI).
2. Railway detects `Dockerfile` / `railway.toml`.
3. Set variables:

| Variable | Example |
|----------|---------|
| `GOOGLE_API_KEY` | your key |
| `CORS_ORIGINS` | `https://your-app.vercel.app,http://localhost:3000` |
| `AUTO_INGEST_ON_START` | `true` (if you bake/ship PDFs in `data/`) |
| `GEMINI_MODEL` | `gemini-2.5-flash` |

4. Optional: attach a **volume** at `/app/storage` (and `/app/data` if uploads should persist).
5. Public URL becomes your API base, e.g. `https://pdf-rag-api.up.railway.app`.

Health check path: `/health`.

---

## Frontend integration (any stack)

```ts
// Example: Next.js / browser fetch
const API = process.env.NEXT_PUBLIC_RAG_API_URL; // e.g. https://xxx.railway.app

// Chat
const res = await fetch(`${API}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "What is Core Technologies?",
    history: [], // optional [{role, content}]
  }),
});
const data = await res.json();
// data.answer, data.sources, data.metrics

// Upload + index PDF
const form = new FormData();
form.append("file", pdfFile);
await fetch(`${API}/ingest/upload`, { method: "POST", body: form });

// Stats
await fetch(`${API}/stats`);
```

Set on Vercel:

```
NEXT_PUBLIC_RAG_API_URL=https://your-railway-url
```

Set on Railway:

```
CORS_ORIGINS=https://your-frontend.vercel.app
```

---

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Index reachable |
| GET | `/stats` | Documents + chunk count |
| POST | `/ingest` | Index `data/*.pdf` (`rebuild`, `force`) |
| POST | `/ingest/upload` | multipart PDF upload + index |
| POST | `/chat` | `{ question, history?, top_k?, min_score?, doc_id? }` |

OpenAPI: `/docs`.

---

## Embeddings note (no torch)

Production uses **fastembed** (ONNX). No PyTorch / torchvision — smaller images and no `torchvision not found` errors.

After changing `EMBEDDING_MODEL`, rebuild the index:

```bash
python app.py ingest --rebuild
# or POST /ingest {"rebuild": true}
```

---

## Checklist before first deploy

- [ ] `GOOGLE_API_KEY` set  
- [ ] `CORS_ORIGINS` limited to your frontend URL(s)  
- [ ] Volume for `/app/storage` if you need persistence  
- [ ] At least one PDF in `data/` **or** upload via API  
- [ ] `GET /health` returns ok  
- [ ] `POST /chat` returns an answer from a separate frontend origin  
