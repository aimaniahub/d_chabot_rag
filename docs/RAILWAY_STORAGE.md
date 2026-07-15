# Railway storage: Volume vs Bucket (for this RAG app)

## Why uploads disappeared after refresh / redeploy

Railway **container disk is temporary**.

Without a **Volume**:

- Files written to `/app/data` or `/app/uploads` are **deleted** on redeploy/restart  
- Chroma under `/app/storage` is **deleted**  
- Admin list looks empty again  

There was also a second bug: on full re-ingest, **prune missing** deleted embeddings when files were gone. That is now **off by default**.

---

## What you need: Volume (required)

Railway docs: **one volume per service**.

### Mount path (must match code)

| Mount path | Contents |
|------------|----------|
| **`/data`** | Everything persistent |

Inside the volume:

```text
/data/uploads/     ← original PDF, DOCX, MD
/data/storage/     ← Chroma vectors + manifest.json + model cache
```

### How to add in Railway UI

1. Open your **RAG service**  
2. **Settings** → **Volumes** (or project **+ New** → Volume)  
3. Attach to this service  
4. **Mount path:** `/data`  
5. Redeploy  

If you previously mounted only `/app/storage`, change it to **`/data`** (or re-upload after switch).

### Env (already in Dockerfile)

```text
PERSIST_ROOT=/data
DATA_DIR=/data/uploads
STORAGE_DIR=/data/storage
PRUNE_MISSING_ON_INGEST=false
```

---

## Railway Buckets (optional, extra safety)

**Buckets** = S3-compatible object storage (private files in the project).

### What buckets are good for

- Backup of **original upload files** (PDF/MD/DOCX)  
- Survive even if someone misconfigures the volume  
- On startup, app can **restore missing files** from the bucket into `/data/uploads` and re-ingest  

### What buckets are **not** ideal for alone

- Chroma vector DB = many small files + local paths  
- Keep **vectors on the Volume** (`/data/storage`)  
- Keep **original files on Volume + optional Bucket copy**

### How to connect a Bucket

1. Railway project → **+ New** → **Bucket** (Storage)  
2. **Connect** / reference it to the RAG service  
3. Railway injects variables such as:

| Variable | Meaning |
|----------|---------|
| `BUCKET` | Bucket name |
| `ACCESS_KEY_ID` | Access key |
| `SECRET_ACCESS_KEY` | Secret |
| `ENDPOINT` | Usually `https://storage.railway.app` |
| `REGION` | Often `auto` |

4. Redeploy  

App auto-detects these and:

- On each successful index → **upload original** to `rag-uploads/<filename>`  
- On startup → **download missing** files back to `/data/uploads`  

---

## Recommended production setup

```text
[Volume @ /data]  →  required for Chroma + local files
[Bucket]          →  optional backup of originals
[Postgres]        →  chat history (already)
```

```text
Upload (admin)
  → save /data/uploads/file.md
  → embed → /data/storage/chroma
  → (optional) copy file to Bucket
Redeploy
  → volume still has files + index
  → if local file missing, restore from Bucket then re-embed
```

---

## Checklist after this deploy

1. Volume mounted at **`/data`**  
2. Redeploy latest image  
3. Admin → Overview: storage warning should clear (or say s3 on)  
4. Bulk-upload the 6 MD files again (one time after volume attach)  
5. Refresh admin — files and chunks should remain  

---

## Admin UI

- **Files → Single / Bulk** upload  
- **Remove** deletes local file + vectors (+ bucket object if enabled)  
- Overview shows **storage warning** if volume looks missing  
