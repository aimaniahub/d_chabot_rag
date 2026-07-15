# Plan + setup: Secure API + Admin UI

## Problem

- Public `/docs` let anyone try chat/upload.
- Need plain answers (no `*`).
- Need a clean place to upload docs and monitor chats.

## Solution

| Piece | Who can use | How |
|-------|-------------|-----|
| `/health` | Public | Liveness only |
| `/admin` | You | Login with **ADMIN_KEY** |
| Upload / re-index | You | Admin UI or `X-API-Key: ADMIN_KEY` |
| `/chat` | Site only | `X-API-Key: API_KEY` (Netlify secret) |
| Public Swagger `/docs` | Off by default | `DOCS_ENABLED=false` |

```
Visitor browser
   → darvigroup.in ChatBot
   → Netlify function (holds RAG_API_KEY secret)
   → Railway /chat + X-API-Key
   → OpenRouter + Chroma + Postgres

You (admin)
   → https://YOUR-APP.up.railway.app/admin
   → enter ADMIN_KEY
   → upload PDF/DOCX, see docs, chats, test Q&A
```

## Railway variables (set all)

```
OPENROUTER_API_KEY=...
OPENROUTER_MODELS=google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,openai/gpt-oss-20b:free
API_KEY=<long random for chat>
ADMIN_KEY=<different long random for admin>
DOCS_ENABLED=false
DATABASE_URL=${{Postgres.DATABASE_URL}}
CORS_ORIGINS=https://darvigroup.in,https://www.darvigroup.in
AUTO_INGEST_ON_START=true
```

## Netlify variables

```
RAG_API_URL=https://dchabotrag-production.up.railway.app
RAG_API_KEY=<same value as Railway API_KEY>
```

## After deploy

1. Open `https://dchabotrag-production.up.railway.app/admin`
2. Log in with **ADMIN_KEY**
3. Upload plant PDFs
4. Use Test chat
5. Site chat works via Netlify (no public upload)

## Answer format (plain text)

```
Guava Varieties
- Royl green
- L-49
Is there anything else you need?
```

No `*` or markdown bold.
