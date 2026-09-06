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

Inference over field photo evidence. Ported from `veritree-survivability`.

| Route | Method | Purpose |
|---|---|---|
| `/computer_vision/survivability_classifier/` | POST | Mangrove detection + alive/dead/unclear classification for an S3 image; writes the annotated image to the destination bucket |
| `/computer_vision/survivability_classifier/image/` | POST | Same, for a directly uploaded file; returns the annotated image |
| `/computer_vision/survivability_classifier/results/` | POST | Same, for a directly uploaded file; returns metrics only |
| `/computer_vision/content_tagging/` | POST | Tag photo evidence for L3 verification (`people`, `meterstick`, ...) |
| `/computer_vision/content_moderation/` | POST | AI content moderation (AICM) over field photos |

`content_tagging` accepts a `provider` field (`anthropic` default, plus `gemini`,
`openai`, `cvmodel`), replacing the four provider-specific routes of the source
service. When it detects `people` it automatically chains into content moderation
and merges the resulting tags, preserving the deployed AICM behaviour.

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

Every route except `/` and `/health` requires a `Token` header matching
`API_ENDPOINT_TOKEN`. The header name is kept from the deployed survivability
service so existing clients work unchanged.

---

## Layout

```
api/
  main.py                  FastAPI app; mounts the two routers
  dependencies.py          shared auth dependency
  routers/
    computer_vision.py
    datascience_results.py
configs/
  config.yaml                        service-level (repo, prod_mode, api_config)
  config_computer_vision.yaml        CV pipeline, L3 verification, AICM prompts
  config_datascience_results.yaml    result-source registry
src/
  main.py                  CLI dispatcher: python src/main.py pipeline pipeline_api
  utils/context.py         config + logging, merges all configs/*.yaml
  awskit/
    datahandlers.py        S3StorageHandler (IAM in prod) + RDSPostgresHandler
    datastores.py          parameterised read-only queries per domain
  libs/
    utils.py               image/EXIF/LLM helpers      (from veritree-survivability)
    veritag.py             OpenCLIP + provider clients (from veritree-survivability)
    geospatial.py          raster helpers, unused for now (from foresthealth)
  services/computer_vision/
    survivability/         inference.py, inference_light.py
    photo_tagging/         verification_photo_tagging.py
  models/                  YOLO detection + Keras classification weights
```

Configuration is split across `configs/*.yaml` and merged into one flat namespace
by `load_config()`, so `config.<key>` works regardless of which file defines it.
A key defined in two files raises at startup.

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env_template .env    # then fill in credentials
```

Run locally:

```bash
python3 src/main.py pipeline pipeline_api
```

Interactive docs at `http://localhost:8000/docs`.

Geospatial extras (`rasterio`, `geopandas`) are optional and excluded from the
default image — see `requirements_geospatial.txt`.

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
| `src/libs/utils.py`, `src/libs/veritag.py`, `src/services/computer_vision/*`, `src/models/*`, `configs/config_computer_vision.yaml`, `Dockerfile` | `veritree-survivability` (branch `aicm`) |
| `src/utils/context.py`, `src/awskit/datahandlers.py`, `src/libs/geospatial.py` | `veritree_multispectral_foresthealth` |
| `src/awskit/datastores.py` | merged from `veritree_bioacoustics`, `veritree_multispectral_foresthealth`, `veritree-tree-tracker-algorithms` |
| `env_template` | `veritree_bioacoustics` |
