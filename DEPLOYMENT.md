# Deployment — Render (backend) + Vercel (frontend)

Two separate deploys from this one repo: the FastAPI backend on Render, the
Vite/React frontend on Vercel. They need each other's URL, so there's a
one-time chicken-and-egg step on the first deploy — see §3.

---

## 1. Backend on Render

### Option A — Blueprint (recommended)

`render.yaml` at the repo root already defines the service (`rootDir: backend`,
Python runtime, health check at `/health`, every env var it needs).

1. Render dashboard → **New** → **Blueprint** → connect this GitHub repo.
2. Render reads `render.yaml` and creates the service. Every var marked
   `sync: false` (the actual secrets) will prompt you to fill it in — nothing
   sensitive lives in the YAML or the repo.
3. Fill in: `GEMINI_API_KEY`, `GROQ_API_KEY` (only needed if `LLM_PROVIDER=groq`),
   `PINECONE_API_KEY`, `APP_API_KEY` (make this a real random secret, not the
   placeholder).
4. Deploy. First build installs `requirements.txt` — no system dependencies
   beyond what pip resolves (Pillow/numpy ship Linux wheels, no extra apt
   packages needed).

### Option B — Manual web service

If not using the Blueprint: **New → Web Service**, connect the repo, then set:

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT --proxy-headers` |
| Health Check Path | `/health` |

Then add the same env vars listed in `render.yaml`'s `envVars` section by hand.

### Option C — Docker

`backend/Dockerfile` also works if you'd rather deploy as a container (Render
supports both). It reads `$PORT` the same way the native runtime does.

### A real limitation: the filesystem is not persistent

Render's default web service disk is **ephemeral** — wiped on every redeploy
and on any restart after the free tier spins down from inactivity. Three
things in this app currently write to local disk:

- `backend/static/catalog/*.jpg` — persisted catalog photos, served back as
  `image_url` in search results.
- `backend/catalog_store.json` — the local mirror used by
  `GET /api/v1/catalog/items` (see `README.md` §7 for why this exists instead
  of reading Pinecone directly).
- The in-memory `_jobs` dict (indexing job status) — not disk-based at all,
  lost on every restart regardless.

None of this breaks *search* — the actual vectors + metadata live in Pinecone,
which is unaffected by a Render restart. What breaks: catalog images already
uploaded will stop resolving (404) after a redeploy, since the files
themselves are gone even though Pinecone still has `image_url` pointing at
them, and the admin "Recently added" table will appear empty until the
catalog is re-indexed. This was true and documented before deployment was in
scope (see `README.md` §7, §11) — deploying to Render just makes the restart
frequency high enough to matter in practice, rather than only "after a
manual server restart."

**If this matters for your use case**, the real fix is moving catalog images
to actual object storage (S3, Cloudflare R2, Render Disks) and
`catalog_store.json`/`_jobs` to a real database (Render Postgres, Redis) —
sized as a follow-up, not done here, since it's a genuine architecture change
rather than a deployment-config one.

---

## 2. Frontend on Vercel

1. Vercel dashboard → **New Project** → import this repo.
2. Set **Root Directory** to `frontend` (Vercel auto-detects Vite once you do).
3. `frontend/vercel.json` already sets the build command, output directory,
   and — importantly — a rewrite rule so every path serves `index.html`.
   **This is required**, not cosmetic: the app has two client-side routes
   (`/` and `/catalog`) via `react-router-dom`, and Vercel's static hosting
   has no server-side knowledge of `/catalog` on its own. Without the
   rewrite, a direct link to or refresh on `/catalog` 404s even though
   clicking the in-app nav link to get there works fine (client-side
   navigation never hits the server).
4. Project Settings → **Environment Variables**, add:
   - `VITE_API_BASE` — the deployed Render backend's URL
     (`https://<your-render-service>.onrender.com`)
   - `VITE_API_KEY` — must match the backend's `APP_API_KEY` exactly
5. Deploy.

**Vite bakes `VITE_*` vars into the built JS at build time**, not read at
runtime — if you change `VITE_API_BASE` or `VITE_API_KEY` in Vercel's
dashboard after the fact, you need to trigger a new deploy for it to take
effect (Vercel does this automatically on redeploy, just not retroactively
on an already-built deployment).

Also worth knowing: because of that same build-time baking, `VITE_API_KEY`
is **not a real secret** once deployed — it's visible in the browser's
network tab and in the bundled JS to anyone who looks. It's enough to stop
casual/accidental use of an otherwise-open URL, not a real access control.
Don't reuse a credential here that actually needs to stay private.

---

## 3. The first-deploy ordering problem

The backend's CORS config needs the frontend's URL; the frontend's
`VITE_API_BASE` needs the backend's URL. Neither is known until the other is
deployed once. Practical order:

1. Deploy the backend first with `ALLOWED_ORIGINS` set to a placeholder (or
   just `http://localhost:5173` temporarily) — it'll come up fine, CORS just
   won't allow the frontend yet.
2. Deploy the frontend, pointing `VITE_API_BASE` at the now-known Render URL.
3. Go back to Render, update `ALLOWED_ORIGINS` to the real Vercel URL (and
   optionally set `ALLOWED_ORIGIN_REGEX` to cover every preview deployment —
   see below), and redeploy/restart the backend for the env var change to
   take effect.

### Vercel preview deployments and CORS

Every branch/PR on Vercel gets its own unique `*.vercel.app` URL. Rather than
adding each one to `ALLOWED_ORIGINS` by hand, set `ALLOWED_ORIGIN_REGEX` on
the backend to match the whole project, e.g. for a Vercel project named
`jewel-thief`:

```text
ALLOWED_ORIGIN_REGEX=https://jewel-thief.*\.vercel\.app
```

This is matched via Starlette's `CORSMiddleware(allow_origin_regex=...)` —
a regex match, evaluated in addition to the exact-match `ALLOWED_ORIGINS`
list, not instead of it.

---

## 4. Verifying it actually works

```bash
# Backend health
curl https://<your-render-service>.onrender.com/health
# -> {"status":"ok"}

# CORS preflight from the real frontend origin
curl -i -X OPTIONS https://<your-render-service>.onrender.com/api/v1/search \
  -H "Origin: https://<your-vercel-app>.vercel.app" \
  -H "Access-Control-Request-Method: POST"
# -> look for Access-Control-Allow-Origin in the response headers
```

Then open the deployed Vercel URL directly at `/catalog` (not by clicking
into it from `/`) — this is the specific case that silently breaks without
the `vercel.json` rewrite rule, so it's worth checking explicitly rather than
only testing in-app navigation.

## 5. What's already handled vs. what's a known gap

**Handled by this deployment setup:**
- Backend binds to Render's injected `$PORT`, not a hardcoded one.
- `/health` wired up for Render's health checks.
- CORS supports both exact origins and a regex, for Vercel's per-branch
  preview URLs.
- SPA client-side routing works on Vercel via the rewrite rule.
- No secrets in the repo — `render.yaml` only stores non-secret config;
  every real credential is `sync: false` (Render prompts, stores encrypted)
  or set directly in Vercel's dashboard.

**Known gaps, not addressed here** (see `README.md` §11 and §1 above for
detail — these are pre-existing architecture tradeoffs, not deployment bugs):
- Catalog images, `catalog_store.json`, and in-memory job status don't
  survive a Render restart/redeploy.
- Single backend instance only — the in-memory job dict and local JSON store
  aren't shared across multiple instances.
- `BackgroundTasks` (in-process) rather than a real task queue — fine at
  current scale, a real constraint if catalog uploads grow large.
