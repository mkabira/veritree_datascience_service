<p align="center">
  <img src="docs/assets/vt_logo.jpeg" alt="veritree" width="180"/>
</p>

# veritree Datascience Service

> One API surface for veritree datascience: computer vision inference over field
> photo evidence, and read access to results published by the analytics pipelines.

This service consolidates work previously spread across three repositories. The
pipelines that *produce* results stay where they are; this repo owns the API that
*serves* them.

---

## Route groups

### `/computer_vision`

One endpoint, dispatching on a `service` field. Ported from `veritree-survivability`.

| Route | Method | Purpose |
|---|---|---|
| `/computer_vision/` | POST | Run the named computer vision service against an image in S3 |

```jsonc
{ "service": "survivability_detection",   // | content_tagging | content_moderation
  "image_url": "bulk_uploads/33/2026-03-16/img.jpg" }
```

| `service` | Does |
|---|---|
| `survivability_detection` | Mangrove detection + alive/dead/unclear classification; returns bounding boxes and per-detection probabilities |
| `content_tagging` | Tags photo evidence for L3 verification (`people`, `meterstick`, ...); chains into moderation when people are present |
| `content_moderation` | AI content moderation (AICM), flagging images that need human review |

`image_url` accepts either a bare **S3 object key** or an **`https://` URL** to the
same object — both forms are in use across the veritree services. The source is read
and decoded **once** per request regardless of service, and never written back to S3
or to local disk, so a photo is stored exactly once.

Every service returns the same envelope; `service` tells the client which `result`
shape it holds:

```jsonc
{
  "service": "survivability_detection",
  "session_id": "…",
  "image_url": "bulk_uploads/33/2026-03-16/img.jpg",
  "image": { "width": 1920, "height": 1080 },
  "result": {
    "counts": { "number_mangroves": 107, "number_alive": 78,
                "number_dead": 4, "number_unclear": 25 },
    "detections": [
      { "detection_id": "…", "index": 0, "status": "alive",
        "detection_confidence": 0.8801,
        "bbox_xyxy":  [854.39, 650.0, 904.03, 807.26],
        "bbox_xywhn": [0.457922, 0.674658, 0.025858, 0.145614],
        "probabilities": { "alive": 0.9999, "dead": 0.0, "unclear": 0.0001 } }
    ]
  }
}
```

`content_tagging` returns `{tags, scores}` and `content_moderation` returns
`{flagged, tags, scores}` in the same `result` slot.

Anthropic is the only tagging backend — it measured best on veritree field photos, so
the OpenCLIP, Gemini and OpenAI taggers were retired. Models load once at import
rather than per request, which with the in-memory path makes survivability detection
**~63% faster per request** (4.26s → 1.60s on a 1920×1080 photo with 107 detections)
for identical predictions.

### `/analyses`

Per-request LLM/model-backed analyses over a payload the caller supplies. Neither
image inference nor a published-table read.

| Route | Method | Purpose |
|---|---|---|
| `/analyses/verification_summarization/` | POST | Convert raw L3 verification rule failures into plain-language flag messages for the field team |

Ported from `veritree-survivability`, where it sits alongside the CV routes despite
never touching an image. **Note the path change from the live service**
(`/verification_summarization/` → `/analyses/verification_summarization/`); clients
need updating at cutover.

### `/datascience_results`

Read-only access to results already written to the analytics Postgres instance.

| Route | Method | Source table | Upstream repo |
|---|---|---|---|
| `/datascience_results/bioacoustics_results/` | GET | `bioacoustics.tbl_postprocess_site_level_indicies` | `veritree_bioacoustics` |
| `/datascience_results/multispectral_results/` | GET | `multispectral.tbl_raster_results` | `veritree_multispectral_foresthealth` |
| `/datascience_results/treetracker_results/` | GET | *pending* | `veritree-tree-tracker-algorithms` |

All three accept filter query params plus `limit`/`offset`. `treetracker_results`
returns **501** until the upstream repo publishes a results table — it currently
writes only staging/fact tables and `dim_*` rollups.

---

## Authentication

Every route in all three groups, except `/` and `/health`, requires a `Token` header matching
`API_ENDPOINT_TOKEN`. The header name is kept from the deployed survivability
service so existing clients work unchanged.

---

## Layout

```
api/
  main.py                  FastAPI app; mounts the routers, serves the favicon
  dependencies.py          shared auth dependency
  static/favicon.ico       veritree mark, generated from docs/assets/vt_logo.jpeg
  routers/
    analyses.py
    computer_vision.py
    datascience_results.py
configs/
  config.yaml                                service-level (repo, prod_mode, api_config)
  config_cv_survivability.yaml               (1) alive/dead detection + classification
  config_cv_content_tagging.yaml             (2) content tagging of photo evidence
  config_cv_content_moderation.yaml          (3) AI content moderation (AICM)
  config_cv_verification_summarization.yaml  (4) planting partners summarization
  config_datascience_results.yaml            result-source registry
scripts/
  generate_favicon.py      regenerates the favicon from the logo
src/
  main.py                  CLI dispatcher: python src/main.py pipeline pipeline_api
  utils/context.py         config + logging, merges all configs/*.yaml
  awskit/
    datahandlers.py        S3 read handler + Postgres engine
    datastores.py          parameterised read-only queries per domain
  libs/
    llm.py                 provider-agnostic llm_chat
    utils.py               crop/resize preprocessing for the survivability classifier
    veritag.py             Anthropic vision classification
  services/computer_vision/
    survivability_detection.py   in-memory detection, models cached once per process
    content_tagging.py           the four tagging backends behind one signature
  models/                  YOLO detection + Keras classification weights
tests/
notebooks/               one API-testing notebook per endpoint
```

Configuration is split across `configs/*.yaml` and merged into one flat namespace
by `load_config()`, so `config.<key>` works regardless of which file defines it.
A key defined in two files raises at startup.

---

## Setup

**Requires Python 3.10 or 3.11** — the Dockerfile uses 3.11, and so should you.

The window is narrow and both ends are hard:

| Bound | Set by | Why |
|---|---|---|
| **>= 3.10** | `src/awskit/datastores.py` | uses PEP 604 `str \| None` annotations, a `TypeError` on 3.9 |
| **<= 3.11** | `tensorflow==2.15.1` | ships wheels only for cp39/cp310/cp311 |

On 3.12+ pip finds no wheel for tensorflow and falls back to building numpy/scipy
from source, which fails without OpenBLAS. If `python3 -V` is outside the window:

```bash
brew install python@3.11
```

Then create the environment with it explicitly:

```bash
/opt/homebrew/bin/python3.11 -m venv venv && source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp env_template .env    # then fill in credentials
```

Run locally:

```bash
python3 src/main.py pipeline pipeline_api
```

Interactive docs at `http://localhost:8000/docs`. Every service is documented there:
click **Authorize**, paste your `API_ENDPOINT_TOKEN` once, then pick a worked example
from the dropdown on `POST /computer_vision/` and hit *Try it out*. `/redoc` renders
the same schema as a reference page.

Geospatial packages (`rasterio`, `geopandas`) are included in the default image;
they pull in GDAL, so expect a correspondingly large build.

---

## Docker

```bash
docker build -t veritree_datascience_service .
docker run --env-file .env -p 8000:8000 veritree_datascience_service
```

---

## Provenance

| Component | Source repo |
|---|---|
| `src/libs/utils.py`, `src/libs/veritag.py`, `src/services/computer_vision/*`, `src/models/*`, `configs/config_cv_*.yaml`, `Dockerfile` | `veritree-survivability` (branch `aicm`) |
| `src/utils/context.py`, `src/awskit/datahandlers.py` | `veritree_multispectral_foresthealth` |
| `src/awskit/datastores.py` | merged from `veritree_bioacoustics`, `veritree_multispectral_foresthealth`, `veritree-tree-tracker-algorithms` |
| `env_template` | `veritree_bioacoustics` |
