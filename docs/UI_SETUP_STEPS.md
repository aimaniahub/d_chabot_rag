# Complete setup steps (Railway UI + Netlify + Darvi site)

Do these **in order**. After each section, use the “Check” row before continuing.

---

## A. Railway — RAG service (already deployed)

### A1. Open the project
1. Go to [railway.app](https://railway.app) → project that has **dchabotrag**.
2. Open the **RAG / API** service (Docker).

### A2. Variables (Variables tab → New Variable)

| Variable | Value |
|----------|--------|
| `OPENROUTER_API_KEY` | Your OpenRouter key (**no quotes**) |
| `OPENROUTER_MODELS` | `google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,openai/gpt-oss-20b:free` |
| `AUTO_INGEST_ON_START` | `true` |
| `CORS_ORIGINS` | `https://darvigroup.in,https://www.darvigroup.in,http://localhost:3000` |

Save → wait for redeploy.

### A3. Volume (persistence)
1. Service → **Settings** → **Volumes** (or project **New** → Volume).
2. Mount path: **`/app/storage`**
3. Redeploy so Chroma index survives restarts.

### A4. Add PostgreSQL (chat history)
1. Project canvas → **+ New** → **Database** → **PostgreSQL**.
2. Open **Postgres** service → **Variables** → copy reference name (usually `DATABASE_URL`).
3. Open **RAG API** service → **Variables** → **Add**:
   - Name: `DATABASE_URL`
   - Value: `${{Postgres.DATABASE_URL}}`  
     (use the variable reference picker if Railway shows “Add variable reference”)
4. Redeploy RAG service.

### A5. (Optional later) Redis
1. **+ New** → **Database** → **Redis**.
2. On RAG service set `REDIS_URL` = `${{Redis.REDIS_URL}}`.  
   (Not required for chat history; useful for rate limits later.)

### A6. Redeploy latest code
1. Ensure GitHub repo `aimaniahub/d_chabot_rag` is connected.
2. **Deploy** latest `main` (Postgres + DOCX support).
3. Wait until deploy is **Success**.

### A7. Checks (browser or PowerShell)

```
https://dchabotrag-production.up.railway.app/health
```
Expect: `"status":"ok"` and `"postgres":{"ok":true,...}` once DB is linked.

```
https://dchabotrag-production.up.railway.app/ready
```
Expect: `chunk_count` > 0.

```
https://dchabotrag-production.up.railway.app/docs
```
→ **POST /chat** → Try it out:
```json
{
  "question": "What is Core Technologies?",
  "session_id": "test-session-1",
  "source": "manual-test",
  "language": "en"
}
```
Expect: `answer` text. If key missing → set `OPENROUTER_API_KEY`.

**Upload company PDF/DOCX (monthly):**
- Swagger → **POST /ingest/upload** → Choose file → Execute  
  or any client with multipart form field `file`.

---

## B. Netlify — darvigroup.in (React)

### B1. Site env vars
1. [Netlify](https://app.netlify.com) → site for **darvigroup.in**.
2. **Site configuration** → **Environment variables** → Add:

| Variable | Value |
|----------|--------|
| `RAG_API_URL` | `https://dchabotrag-production.up.railway.app` |

(Optional local `.env` / `.env.local` same key.)

### B2. Deploy frontend code
Deploy the updated `D:\darvi-registration` code that includes:
- `netlify/functions/gemini-chat.js` (proxy to Railway)
- `src/components/ChatBot.tsx` (re-enabled + session id)
- `src/components/Layout.tsx` (renders `<ChatBot />`)

```bash
cd D:\darvi-registration
# commit & push your usual Netlify git remote, or:
netlify deploy --prod
```

### B3. Checks
1. Open https://darvigroup.in  
2. Wait ~3s or click the green chat FAB (bottom-right).  
3. Ask: “What services does Darvi provide?”  
4. Expect an answer (not only error).  
5. On Railway Postgres → **Data** tab: rows in `messages` / `conversations`.

---

## C. Monthly knowledge update (you)

1. Prepare new **PDF or DOCX** (company details).  
2. Open Railway API `/docs` → **POST /ingest/upload**.  
3. Upload file.  
4. Check `/stats` — document list includes new file.  
5. Ask a related question on the site.  

No site redeploy needed for document updates.

---

## D. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/chat` → Missing OPENROUTER_API_KEY | Set variable on **RAG** service, redeploy |
| `postgres.ok: false` | Wire `DATABASE_URL=${{Postgres.DATABASE_URL}}` |
| Site chat timeout / error | Confirm Netlify `RAG_API_URL`; check Railway logs |
| Empty index | `/ingest` or upload a PDF; enable `AUTO_INGEST_ON_START` |
| CORS errors in browser console | You should call **Netlify function**, not Railway from browser. If calling Railway direct, set `CORS_ORIGINS` |

---

## E. Architecture reminder

```
Visitor → ChatBot (React)
       → /.netlify/functions/gemini-chat
       → Railway POST /chat
       → Chroma + Gemini
       → Postgres (Q&A log)
```
