# Connect your S3 bucket (Tigris / Railway)

Your credentials panel values map to Railway **Variables** like this:

| UI label | Railway variable name | Your value |
|----------|----------------------|------------|
| Endpoint URL | `ENDPOINT` or `AWS_ENDPOINT_URL` | `https://t3.storageapi.dev` |
| Region | `REGION` or `AWS_REGION` | `auto` |
| Bucket Name | `BUCKET` or `S3_BUCKET` | `recorded-gyoza-qmxdifcure` |
| Access Key ID | `ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID` | `tid_...` |
| Secret Access Key | `SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY` | *(paste secret)* |

Also set:

```text
S3_PREFIX=rag-uploads/
```

## Railway UI steps

1. Open **RAG service** → **Variables**
2. Add (or confirm “Add to Service” already injected them):
   - `ENDPOINT` = `https://t3.storageapi.dev`
   - `REGION` = `auto`
   - `BUCKET` = `recorded-gyoza-qmxdifcure`
   - `ACCESS_KEY_ID` = your access key
   - `SECRET_ACCESS_KEY` = your secret
   - `S3_PREFIX` = `rag-uploads/`
3. Still required for embeddings: **Volume mounted at `/data`**
4. Redeploy

## Verify in Admin

1. Open `/admin` → login  
2. **Files** tab → **Test bucket**  
3. Expect `"ok": true` and `object_count`  
4. Bulk upload MD files again  
5. Each file is saved under `/data/uploads` **and** copied to the bucket  

## What bucket does vs volume

- **Volume `/data`**: live files + Chroma vectors (required for chat)  
- **Bucket**: durable copy of originals; restore if local files missing  

## Security

Do not commit keys to Git. If keys were shared in chat, rotate them in the bucket console when possible.
