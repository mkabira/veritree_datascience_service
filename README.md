<p align="center">
  <img src="docs/assets/vt_logo.jpeg" alt="veritree" width="180"/>
</p>

# veritree Datascience Service

> One API for veritree datascience: computer vision over field photo evidence, and
> read access to the results the analytics pipelines produce — from the database, or
> by scanning S3 directly.

This service consolidates work that previously lived in separate repositories. The
pipelines that **produce** results stay where they are; this repo owns the API that
**serves** them, plus the computer vision models that run per request.

---

## Contents

- [At a glance](#at-a-glance)
- [Quick start](#quick-start)
- [API reference](#api-reference)
  - [Authentication](#authentication)
  - [`POST /computer_vision/`](#post-computer_vision)
  - [`GET /datascience_results/…`](#get-datascience_results)
  - [`POST /analyses/verification_summarization/`](#post-analysesverification_summarization)
  - [Service endpoints](#service-endpoints)
  - [Status codes](#status-codes)
- [Configuration](#configuration)
  - [Environment variables](#environment-variables)
  - [Config files](#config-files)
  - [Connecting through an SSH bastion](#connecting-through-an-ssh-bastion)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [Provenance](#provenance)

---

## At a glance

| | |
|---|---|
| **Runtime** | Python 3.10–3.11, FastAPI, uvicorn |
| **Endpoints** | 4 functional + 2 service |
| **Models** | YOLO (detection), Keras (classification), Anthropic (vision tagging) |
| **Data sources** | Postgres (per domain), S3 live scan |
| **Tests** | 185, no credentials or network required |
| **Interactive docs** | `/docs` (Swagger) and `/redoc` |

---

## Quick start

**Requires Python 3.10 or 3.11.** The window is narrow and both ends are hard:

| Bound | Set by | Why |
|---|---|---|
| **>= 3.10** | `src/awskit/datastores.py` | uses PEP 604 `str \| None` annotations, a `TypeError` on 3.9 |
| **<= 3.11** | `tensorflow==2.15.1` | ships wheels only for cp39/cp310/cp311 |

On 3.12+ pip finds no TensorFlow wheel and falls back to building numpy/scipy from
source, which fails without OpenBLAS.

```bash
python3.11 -m venv venv && source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp env_template .env    # then fill in credentials
```

```bash
python3 src/main.py pipeline pipeline_api
```

Then open **http://localhost:8000/docs**, click **Authorize**, paste your
`API_ENDPOINT_TOKEN`, and use *Try it out* — the `POST /computer_vision/` body carries
a worked example per service.

> **First start is slow.** The Keras and YOLO weights load at import, before the server
> binds its port. It looks like a hang; it isn't.

---

## API reference

### Authentication

Every route except `/` and `/health` requires a `Token` header:

```http
Token: <API_ENDPOINT_TOKEN>
```

A wrong token returns `401` — that is this service's own check. A **missing** header is
rejected by FastAPI's `APIKeyHeader` before the check runs, and its status is a
framework detail: `403` on the pinned `fastapi==0.115.11`, `401` from roughly 0.14x
onward. Treat any of `401`/`403` as unauthenticated rather than branching on the exact
code.

---

### `POST /computer_vision/`

One endpoint, three model pipelines, selected by a **`service`** field.

```jsonc
{
  "service": "survivability_detection",   // | content_tagging | content_moderation
  "image_url": "s3://veritree-impactteam-survivability/survivability/100/img.jpeg"
}
```

| `service` | Does |
|---|---|
| `survivability_detection` | Detects mangroves and classifies each as alive, dead or unclear. Returns bounding boxes and per-detection probabilities so the client draws its own overlay. |
| `content_tagging` | Tags photo evidence for L3 verification (`people`, `meterstick`, …). Chains into moderation when people are present. |
| `content_moderation` | Flags images needing human review before publication. A triage layer, never the final decision. |

**Addressing the image.** `image_url` accepts three forms, all in use across the
veritree services:

| Form | Read via | Notes |
|---|---|---|
| `s3://bucket/key` | boto3 | names its own bucket, works with no default configured |
| `survivability/100/img.jpg` | boto3 | bare object key, resolved against the default source bucket |
| `https://…` | HTTP | the object must be reachable over HTTP |

The image is read and decoded **once per request** whichever service runs, and is never
written back to S3 or to local disk — a photo is stored exactly once.

**Response envelope.** Identical for every service; `service` tells you which `result`
shape you have:

```jsonc
{
  "service": "survivability_detection",
  "session_id": "4cd88ae5-…",
  "image_url": "s3://…/img.jpeg",
  "image": { "width": 1920, "height": 1080 },
  "result": {
    "counts": { "number_mangroves": 107, "number_alive": 78,
                "number_dead": 4, "number_unclear": 25 },
    "detections": [
      { "detection_id": "…", "index": 0, "status": "alive",
        "detection_confidence": 0.8801,
        "bbox_xyxy":  [854.39, 650.0, 904.03, 807.26],   // absolute pixels
        "bbox_xywhn": [0.457922, 0.674658, 0.025858, 0.145614],  // normalised
        "probabilities": { "alive": 0.9999, "dead": 0.0, "unclear": 0.0001 } }
    ]
  }
}
```

`content_tagging` returns `{tags, scores}` and `content_moderation` returns
`{flagged, tags, scores}` in the same `result` slot. `scores` carries **every** candidate
tag, not just those above threshold — useful for tuning thresholds without redeploying.

**Chaining.** When `content_tagging` detects `people` it routes the image through
content moderation automatically and merges those tags into its own, so a tagging call
can legitimately return moderation tags.

**Anthropic is the only tagging backend** — it measured best on veritree field photos,
so the OpenCLIP, Gemini and OpenAI taggers were retired and there is no `provider` field.

---

### `GET /datascience_results/…`

Results the upstream pipelines produce. These routes compute nothing.

| Route | Source | Filters |
|---|---|---|
| `bioacoustics_results` | analytics database | `code_country`, `code_site`, `recording_year` |
| `multispectral_results` | **`live_scan=true`** — S3 | `country`, `site`, `subsite`, `period`, `raster` |
| `treetracker_results` | **`live_scan=true`** — S3 | `session_id` (`country`/`site` are accepted for parity but ignored — S3 keys carry no such fields) |

All take `limit` (1–5000, default 1000) and `offset`. Filters are optional; omitting one
widens the result.

```bash
curl "$BASE/datascience_results/bioacoustics_results/?code_site=site_nicola&limit=50" -H "Token: $TOKEN"
curl "$BASE/datascience_results/multispectral_results/?live_scan=true&country=kenya&raster=PRODUCTIVITY" -H "Token: $TOKEN"
curl "$BASE/datascience_results/treetracker_results/?live_scan=true&session_id=10691" -H "Token: $TOKEN"
```

**Shared envelope:**

```jsonc
{
  "source_type": "database",        // or "live_scan"
  "source": "bioacoustics.tbl_postprocess_site_level_indicies",
  "n_records": 200,                 // total matching the filters
  "n_returned": 50,                 // rows in this page
  "limit": 50, "offset": 0,
  "results": [ /* … */ ]
}
```

**Live scan.** `multispectral_results` and `treetracker_results` index their assets
straight out of S3 rather than reading a published table, so a result appears as soon as
the pipeline writes it — no publish step in between. Every field is derived from the
object keys. Without `live_scan=true` they return **501**, because their database
sources are not implemented.

A scan lists a whole prefix, so its cost grows with the bucket rather than the result.
Each domain has a **5-minute budget** (`timeout_seconds`); exceeding it returns **504**
rather than a partial index, since half a listing looks complete to the caller.

**Paging.** Database reads apply `LIMIT`/`OFFSET` in SQL alongside a `COUNT(*)`, so a
response is bounded by the page rather than the table. Live scans page in Python — S3
has no equivalent, and the prefix is listed either way.

---

### `POST /analyses/verification_summarization/`

Converts raw L3 verification rule failures into plain-language flag messages for the
field team. Text in, text out — it never touches an image, which is why it sits under
`/analyses` rather than `/computer_vision`.

```jsonc
{
  "rules": [
    { "rule_public_id": "rule_abc123",
      "name": "Photo missing meterstick",
      "status": "failed",
      "failure_reason": { "detail": "no meterstick detected in 4 of 12 photos" },
      "flagged": true,
      "comments": [ { "source": "field", "actor_id": "u1",
                      "message": "stick was out of frame", "created_at": "2026-03-16" } ] }
  ]
}
```

Returns `{session_id, suggested_flags: [{rule_public_id, message}]}`. Rule comments are
included in the prompt, so field-team context reaches the model. The model is asked for
a JSON array; malformed output surfaces as a **500** rather than a partial response, so
a bad generation is never mistaken for "no flags".

The provider is set by `model` in `configs/config_cv_verification_summarization.yaml`
(`claude` by default) and routed through litellm, so `gpt` and `gemini` aliases work too.

---

### Service endpoints

Unauthenticated.

| Route | Returns |
|---|---|
| `GET /` | Service banner |
| `GET /health` | Liveness probe for the ECS target group |

```json
{"status": "ok", "service": "veritree_datascience_service",
 "version": "0.1.0", "timestamp": "2026-09-08T00:52:20+00:00"}
```

`timestamp` is an ISO-8601 instant carrying its UTC offset, so a reader can convert it
without knowing where the task ran.

---

### Status codes

| Code | Means |
|---|---|
| `400` | The request pointed at something unusable — an S3 object that is not an image, a malformed `s3://` URI |
| `401` | Missing or invalid `Token` header |
| `422` | A field failed validation — an unknown `service`, a `limit` out of range |
| `500` | The service itself failed |
| `501` | That source is not available here — not implemented, or not published upstream |
| `502` | An upstream store (S3 or the database) failed |
| `504` | A live S3 scan exceeded its time budget |

Errors never return the underlying SQL or S3 message; those go to the logs. `501` is
deliberately distinct from `500`: "not available here" rather than a fault.

---

## Configuration

### Environment variables

Copy `env_template` to `.env`. `src/utils/context.py` loads it at import.

| Group | Variables |
|---|---|
| **AWS account** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_REGION_NAME`, `AWS_S3_BUCKET_NAME` |
| **SSH bastion** (shared) | `AWS_RDS_SSH_HOST`, `AWS_RDS_SSH_PORT`, `AWS_RDS_SSH_USER`, `AWS_RDS_SSH_PKEY`, `AWS_RDS_SSH_PKEY_PASSWORD` |
| **Per-domain database** | `AWS_RDS_<DOMAIN>_{ENDPOINT,PORT,DB,USER_NAME,PASSWORD}` |
| **API** | `API_ENDPOINT_TOKEN` |
| **Model providers** | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_AI_STUDIO_API_KEY` |

Two rules govern resolution:

- **Database settings have no shared fallback.** `<DOMAIN>` is `BIOACOUSTICS`,
  `MULTISPECTRAL` or `TREETRACKER`. A domain is unconfigured unless its own variables
  are set, and its route returns `501` naming the ones missing. Falling back would point
  a domain at another domain's server and query an instance that does not hold its schema.
- **SSH settings do fall back**, from `AWS_RDS_<DOMAIN>_SSH_*` to the shared
  `AWS_RDS_SSH_*`. One bastion usually fronts every database, so repeating the host and
  key per domain is noise.

> `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are never read by an `os.getenv()`
> call — boto3's default credential chain picks them up from the environment, which is
> what a bare `boto3.Session()` resolves against when `prod_mode` is true. **Do not
> remove them because a code search finds no reference.**

### Config files

Every `configs/*.yaml` is merged into one flat namespace at import, so `config.<key>`
works regardless of which file defines it. A key defined in two files raises at startup.

| File | Owns |
|---|---|
| `config.yaml` | `prod_mode`, `repo`, `api_config`, `logging` |
| `config_cv_survivability.yaml` | detection/classification pipeline, class map |
| `config_cv_content_tagging.yaml` | tagging model, prompts, tags, threshold |
| `config_cv_content_moderation.yaml` | AICM model, prompt, calibrated threshold |
| `config_cv_verification_summarization.yaml` | summarization model and prompts |
| `config_datascience_results.yaml` | result-source registry and `live_scan` settings |

`prod_mode` selects how S3 authenticates: `true` uses the ambient credential chain (the
ECS task role); `false` uses the explicit keys. **Set it to `false` locally**, or your
`.env` keys are ignored.

### Connecting through an SSH bastion

If a database is only reachable via a jump host, set `AWS_RDS_SSH_HOST` and the service
forwards every connection through it. Leave it unset to connect directly.

```
AWS_RDS_SSH_HOST = "bastion.example.com"
AWS_RDS_SSH_PORT = "22"
AWS_RDS_SSH_USER = "ec2-user"
AWS_RDS_SSH_PKEY = "~/.ssh/id_rsa"
AWS_RDS_SSH_PKEY_PASSWORD = ""       # only if the key is encrypted
```

`AWS_RDS_<DOMAIN>_ENDPOINT` stays the database's own hostname — it is resolved from the
bastion's side of the forward, not from your machine. The local port is chosen by the OS
so concurrent workers don't collide, and is re-resolved on every connection, so a tunnel
that drops and restarts on a different port keeps working. Restarts are serialised
behind a lock, since handlers run concurrently in a threadpool.

> **`sshtunnel` requires `paramiko < 4`.** Its latest release references
> `paramiko.DSSKey`, removed in paramiko 4.0 with DSA support. `requirements.txt` pins
> it; an unpinned install picks up paramiko 5 and every tunnel dies with `AttributeError`.

---

## Architecture

```
                        ┌──────────────────────────────┐
   client ──Token──────▶│  FastAPI  (api/)             │
                        │  routers: computer_vision,   │
                        │  datascience_results,        │
                        │  analyses                    │
                        └───────┬──────────┬───────────┘
                                │          │
             ┌──────────────────┘          └───────────────────┐
             ▼                                                 ▼
   ┌────────────────────┐                        ┌──────────────────────────┐
   │ src/services/      │                        │ src/awskit/              │
   │  survivability_    │                        │  datastores  (queries)   │
   │   detection (YOLO  │                        │  datahandlers (engines,  │
   │   + Keras)         │                        │   S3, SSH tunnel)        │
   │  content_tagging   │                        │ src/services/live_scan   │
   │   (Anthropic)      │                        │   (S3 indexing)          │
   └─────────┬──────────┘                        └────────┬─────────────────┘
             ▼                                            ▼
        S3 (images)                          Postgres per domain  │  S3 (assets)
```

**Request handling.** Handlers that perform blocking I/O are declared `def`, not
`async def`, so FastAPI runs them in a threadpool and the event loop stays free. The
same body in an `async def` stalls every concurrent request — measured at **1.28s** for
`/health` during a single inference, versus **1.6ms** now. Trivial handlers (`/`,
`/health`, favicon, docs) stay `async`, avoiding a needless threadpool hop.

**Model lifecycle.** YOLO, Keras and the Anthropic client load once at import, not per
request. With the in-memory image path this makes survivability detection **~63% faster
per request** (4.26s → 1.60s on a 1920×1080 photo with 107 detections) for identical
predictions.

**Database connections.** One pooled SQLAlchemy engine per domain, built lazily and
cached — an unused domain never opens a connection or an SSH tunnel. Pools use
`pool_pre_ping` and `pool_recycle=1800` because RDS drops idle connections, a 10s
connect timeout and a 30s server-side `statement_timeout`. Credentials go through
`URL.create()`, so a password containing `@`, `/`, `:` or `#` connects correctly and the
URL masks the password when logged.

**Logging.** Every line reads `<component> <event>` followed by `key=value` pairs, so
lines are greppable without a structured-logging dependency:

```
computer_vision survivability_detection started: session_id=4a8f99e0 image=s3://…
survivability_detection finished: session_id=4a8f99e0 mangroves=14 alive=14 dead=0 unclear=0 duration=1.13s
computer_vision survivability_detection finished: session_id=4a8f99e0 duration=2.38s
```

Request-scoped lines carry a correlation id, so one request can be followed through an
interleaved log — necessary now that handlers run concurrently in a threadpool.
`session_id` is the id returned to the client by `/computer_vision/` and `/analyses/…`;
`request_id` is the internal equivalent on `/datascience_results/…`, which returns no
such field and whose `session_id` filter already means the planting session.

Levels carry meaning: `info` for lifecycle and outcomes, `warning` for a
degraded-but-handled condition, `error` for a failed request. Nothing routine logs above
`info`. Credentials never reach the log — database URLs render through SQLAlchemy, which
masks the password.

Output goes to a size-rotated file, 10 MB × 5 backups — a 60 MB ceiling — plus stdout
for CloudWatch. Configurable under `logging:` in `config.yaml`. The convention is
documented in `src/utils/context.py` and enforced by `tests/test_logging.py`.

---

## Repository layout

```
api/
  main.py                  FastAPI app, OpenAPI metadata, service endpoints
  dependencies.py          shared Token auth dependency
  static/favicon.ico       veritree mark, generated from docs/assets/vt_logo.jpeg
  routers/
    computer_vision.py     one endpoint, three services
    datascience_results.py database and live-scan result routes
    analyses.py            verification summarization
configs/                   see Config files above
notebooks/                 one API-testing notebook per endpoint
scripts/
  generate_favicon.py      regenerates the favicon from the logo
src/
  main.py                  CLI dispatcher: python src/main.py pipeline pipeline_api
  utils/context.py         config merge + rotating logger
  awskit/
    datahandlers.py        per-domain engines, SSH tunnel, S3 reader
    datastores.py          parameterised read-only queries
  libs/
    llm.py                 provider-agnostic llm_chat
    utils.py               crop/resize preprocessing for the classifier
    veritag.py             Anthropic vision classification
  services/
    live_scan.py           S3 indexing for multispectral and treetracker
    computer_vision/
      survivability_detection.py   in-memory detection, models cached
      content_tagging.py           Anthropic tagging and moderation
  models/                  YOLO detection + Keras classification weights
tests/                     185 tests, no credentials or network required
```

---

## Development

**Run the tests** — no install of the CV stack, no credentials, no network:

```bash
pytest tests/ -q
```

The suite strips every `AWS_*` variable at import, so no test can reach live
infrastructure regardless of what is in your `.env`. Model modules are stubbed in
`conftest.py`, and database reads run against a file-backed sqlite fixture.

Beyond route behaviour, the suite pins the properties that fail silently:

| Guard | Catches |
|---|---|
| `test_concurrency.py` | a blocking handler declared `async`, and unserialised tunnel restarts |
| `test_repo_hygiene.py` | an imported package missing from `requirements.txt`, a file with no trailing newline, credentials not gitignored |
| `test_live_scan.py` | an unbounded scan |
| `test_context.py` | a config key missing, duplicated, or a dead one creeping back |

**Iterate with auto-reload**, skipping the CLI dispatcher:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Notebooks.** One per endpoint under `notebooks/`, each with setup, a health check, an
editable request, the raw response, and the error cases worth confirming. Clear outputs
before committing:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## Deployment

```bash
docker build -t veritree_datascience_service .
```

```bash
docker run --env-file .env -p 8000:8000 veritree_datascience_service
```

The image is Python 3.11-slim, multi-stage, and runs
`python3 src/main.py pipeline pipeline_api` — the task/pipeline contract the ECS task
definitions invoke, shared with the other veritree datascience repositories.

Point the target group's health check at `/health`.

---

## Known limitations

These are real and worth knowing before a production rollout.

1. **The SSH key is not in the image.** The Dockerfile copies no key and nothing mounts
   one, so `AWS_RDS_SSH_PKEY` has no file to read inside the container — the tunnel
   works locally only. Mount it as a secret, or prefer network-level access (VPC
   peering, RDS Proxy, or an `ssh -L` sidecar). An in-process tunnel also means each
   uvicorn worker opens its own SSH session against the bastion.

2. **`src/main.py` launches uvicorn with `subprocess.run`,** so SIGTERM reaches the
   Python parent and never the server. On deploy the parent dies, uvicorn keeps serving
   orphaned, and ECS force-kills after the stop timeout instead of draining. Replacing
   `subprocess.run` with `os.execvp` fixes it and keeps the task-name contract.

3. **Live scans are uncached and list the whole prefix.** Filters and paging apply after
   the listing, so a narrow query costs the same as a broad one. Fine at 87 rasters and
   400 tracker assets; before those grow, push filters into the S3 `Prefix` and add a
   short TTL cache.

4. **`read_properties` on tree-tracker scans is an N+1** — one extra S3 GET per asset.
   Off by default, which is right.

5. **Database sources for `multispectral_results` and `treetracker_results` are not
   implemented.** Both answer from S3. The accessors in `src/awskit/datastores.py` carry
   the wiring instructions for restoring them.

---

## Provenance

| Component | Source repository |
|---|---|
| `src/libs/utils.py`, `src/libs/veritag.py`, `src/services/computer_vision/*`, `src/models/*`, `configs/config_cv_*.yaml`, `Dockerfile` | `veritree-survivability` (branch `aicm`) |
| `src/utils/context.py`, `src/awskit/datahandlers.py` | `veritree_multispectral_foresthealth` |
| `src/awskit/datastores.py` | merged from `veritree_bioacoustics`, `veritree_multispectral_foresthealth`, `veritree-tree-tracker-algorithms` |
| `src/services/live_scan.py` | ported from the indexing tasks in `veritree_multispectral_foresthealth` and `veritree-tree-tracker-algorithms` |

Two configuration errors in the upstream repositories were corrected here and verified
against the live buckets: the multispectral rasters are in
`veritree-impactteam-multispectral` (not `veritree-business-intelligence`), and
`veritree-tree-trackers` is in `us-east-2` (not `ca-central-1`).

---

<p align="center"><sub>veritree · Data Science</sub></p>
