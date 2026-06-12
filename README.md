---
title: PDF Agent Backend
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# PDF Agent — Backend (FastAPI)

AI-powered PDF agent: upload → classify → summarize → chat (RAG) with citations.

This repository is configured to run as a **Hugging Face Space (Docker SDK)**.
HF builds the `Dockerfile` at the repo root and routes traffic to `app_port`
(8000, matching the container's `uvicorn` port). The free CPU Space gives 16 GB
RAM / 2 vCPU — enough headroom for the embedding model, unlike 512 MB tiers.

## Required Space secrets

Set these under **Settings → Variables and secrets** (never commit them):

| Key | Notes |
| --- | --- |
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_JWT_SECRET` | **Mandatory** — auth fails closed without it |
| `SUPABASE_SERVICE_ROLE_KEY` | Admin endpoints only |
| `DATABASE_URL` | Supabase Postgres connection string (recommended over SQLite) |
| `LLM_PROVIDER` | e.g. `gemini` |
| `GEMINI_API_KEY` | (or the key for whichever provider you choose) |
| `CORS_ALLOWED_ORIGINS` | Your frontend origin, e.g. `https://pdf-agent-xxxx.vercel.app` (no trailing slash) |
| `CORS_ALLOWED_ORIGIN_REGEX` | Optional, for Vercel previews: `^https://pdf-agent-[a-z0-9-]+\.vercel\.app$` |

## Storage note (free tier)

The free Space's disk is **ephemeral** — it resets on rebuild/restart. Keep the
relational data in Supabase Postgres (`DATABASE_URL`). The Chroma vector store
under `data/vectorstore` will not persist, so documents need to be re-indexed
after a restart (this happens automatically as documents are re-uploaded). For
durable vectors, mount persistent storage (paid) or use a hosted vector DB.

## Local / other platforms

The same image runs via `docker compose up` (port 8000) and on Render
(`render.yaml`, which overrides the start command with `$PORT`). See
`DOCKER_SETUP.md` and `SUPABASE_SETUP.md` for details.

## Health

`GET /health` → `{"status": "ok"}`. API docs at `/docs`.
