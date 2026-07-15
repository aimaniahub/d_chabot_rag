# Deploy to Railway

This backend deploys as a **Docker service** on Railway.

## 1. Prerequisites

- GitHub repo: `https://github.com/aimaniahub/d_chabot_rag`
- Railway account: https://railway.app
- Gemini API key (`GOOGLE_API_KEY`)

## 2. Deploy from GitHub (dashboard)

1. Open [Railway New Project](https://railway.app/new)
2. **Deploy from GitHub repo** → select `aimaniahub/d_chabot_rag`
3. Railway detects `Dockerfile` + `railway.toml`
4. Open the service → **Variables** → add:

| Variable | Value |
|----------|--------|
| `GOOGLE_API_KEY` | your Gemini key |
| `AUTO_INGEST_ON_START` | `true` |
| `CORS_ORIGINS` | `*` or your Vercel URL(s), comma-separated |
| `GEMINI_MODEL` | `gemini-2.5-flash` (optional) |

5. **Settings → Networking → Generate Domain** (public HTTPS URL)
6. **Volumes** (recommended for persistence):
   - Mount path: `/app/storage`
   - Keeps Chroma index + model cache across restarts

7. Deploy / wait for build → open `https://YOUR-APP.up.railway.app/health`

## 3. Deploy from CLI

```bash
npm i -g @railway/cli
railway login
railway init          # or: railway link
railway variables set GOOGLE_API_KEY=your_key
railway variables set AUTO_INGEST_ON_START=true
railway variables set CORS_ORIGINS=*
railway up
railway domain
```

## 4. Verify

```bash
curl https://YOUR-APP.up.railway.app/health
curl https://YOUR-APP.up.railway.app/ready
curl -X POST https://YOUR-APP.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is Core Technologies?\"}"
```

## 5. Connect frontend (Vercel)

```env
NEXT_PUBLIC_RAG_API_URL=https://YOUR-APP.up.railway.app
```

Railway:

```env
CORS_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

## 6. Notes

- First deploy may take several minutes (Docker build + embedding model download).
- Health is `/health` (fast). Index builds in background when `AUTO_INGEST_ON_START=true`.
- Without a volume on `/app/storage`, the vector index resets on every new deploy.
- Do **not** put secrets in git; only set them in Railway Variables.
