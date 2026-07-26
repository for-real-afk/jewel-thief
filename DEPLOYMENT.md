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
| --- | --- |
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers` |
| Health Check Path | `/health` |

Then add the same env vars listed in `render.yaml`'s `envVars` section by hand.

### Option C — Docker

`backend/Dockerfile` also works if you'd rather deploy as a container (Render
supports both). It reads `$PORT` the same way the native runtime does.

### A real limitation: the filesystem is not persistent

Render's default web service disk is **ephemeral** — wiped on every redeploy
and on any restart after the free tier spins down from inactivity. This bit
in practice, twice: catalog images written at indexing time (via
`POST /api/v1/catalog/index`) vanished after the next restart, breaking
every thumbnail even though search itself kept working fine (the vectors +
`image_url` metadata live in Pinecone, unaffected by a Render restart —
only the actual image *files* were gone).

**Current fix**: `backend/app/static/catalog/*.jpg` (the full image set at time
of writing) and `backend/catalog_store.json` are committed directly into the
repo instead of being left as runtime-only state — see the "Catalog images
are committed to the repo" note in `README.md` §7. They now ship with every
deploy regardless of restarts.

**This does not fully solve the underlying issue** — it only covers
whatever was committed. Any *new* item added through the live `/catalog`
admin page after that commit is still written only to the running
container's disk, and is exactly as ephemeral as before: it'll disappear on
the next restart unless someone commits it too (or reruns
`backend/scripts/reindex_to_remote.py` against the live URL after a
restart, which rewrites the same runtime state and has the same limitation).
The in-memory `_jobs` dict (indexing job status) was never disk-based at all
and is lost on every restart regardless of any of this.

**The actual permanent fix**, if the catalog is expected to grow via the
admin UI in normal operation (not just redeploy the current fixed set):
move catalog images to real object storage (S3, Cloudflare R2, Render
Disks) and `catalog_store.json`/`_jobs` to a real database (Render
Postgres, Redis). Sized as a follow-up, not done here — it's a genuine
architecture change, not a deployment-config one.

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
- The catalog image set at time of writing is committed into the repo
  (`backend/app/static/catalog/`), so it survives Render restarts — see §1 above.

**Known gaps, not addressed here** (see `README.md` §11 and §1 above for
detail — these are pre-existing architecture tradeoffs, not deployment bugs):

- Any catalog item added through the live `/catalog` admin page *after* the
  last commit of `backend/app/static/catalog/` is still runtime-only and does
  not survive a restart/redeploy — only the committed set does.
- The in-memory `_jobs` dict (indexing job status) is never disk-based and
  is lost on every restart regardless.
- Single backend instance only — the in-memory job dict and local JSON store
  aren't shared across multiple instances.
- `BackgroundTasks` (in-process) rather than a real task queue — fine at
  current scale, a real constraint if catalog uploads grow large.
