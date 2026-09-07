"""
veritree datascience service -- FastAPI application entrypoint.

Route groups:
  /computer_vision      inference over field photo evidence
  /datascience_results  read-only results from the analytics database
  /analyses             per-request LLM/model-backed analyses over a supplied payload
"""

import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse

from api.routers import analyses, computer_vision, datascience_results

from src.utils import context


config = context.config
logger = context.logger


FAVICON_PATH = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')


# Rendered above the route list in /docs. Markdown is supported.
API_DESCRIPTION = """
Unified API for veritree datascience: computer vision over field photo evidence, and
read access to results published by the analytics pipelines.

## Authentication

Every route except `/` and `/health` requires a `Token` header:

```
Token: <API_ENDPOINT_TOKEN>
```

In this page, click **Authorize** and paste the token once to have it sent with every
*Try it out* request.

## Route groups

| Group | What it does |
|---|---|
| **computer_vision** | Runs a model over one image in S3. One endpoint; the `service` field selects the pipeline. |
| **datascience_results** | Serves rows the upstream pipelines already published. Filtered and paged by the database. |
| **analyses** | Per-request LLM analyses over a payload you supply. No image involved. |
"""

TAGS_METADATA = [
    {
        "name": "computer_vision",
        "description": (
            "Model inference over a single field photo. One endpoint routes to three "
            "services via the **`service`** field:\n\n"
            "- **`survivability_detection`** — detects mangroves and classifies each as "
            "alive, dead or unclear. Returns bounding boxes and per-detection "
            "probabilities so the client draws its own overlay.\n"
            "- **`content_tagging`** — tags photo evidence for L3 verification "
            "(`people`, `meterstick`, ...). Chains into moderation when people are present.\n"
            "- **`content_moderation`** — flags images needing human review before "
            "publication. A triage layer, never the final decision.\n\n"
            "`image_url` accepts a bare **S3 object key** or an **`https://` URL** to the "
            "same object. The image is read and decoded once per request and never "
            "written back, so a photo is stored exactly once."
        ),
    },
    {
        "name": "datascience_results",
        "description": (
            "Read-only access to results already written to the analytics database. "
            "These routes compute nothing.\n\n"
            "All filters are optional; `limit` and `offset` are applied by the database, "
            "so a response is bounded by the page rather than the table. `n_records` is "
            "the total matching the filters, `n_returned` the size of this page."
        ),
    },
    {
        "name": "analyses",
        "description": (
            "Per-request LLM- or model-backed analyses over a payload the caller supplies. "
            "Distinct from **computer_vision** (which needs an image) and "
            "**datascience_results** (which serves published rows)."
        ),
    },
]


# The default docs routes hardcode FastAPI's own favicon, so they are disabled here
# and re-declared below against ours. openapi_url is left at its default.
app = FastAPI(
    title=config.repo.name,
    description=API_DESCRIPTION,
    summary=config.repo.description,
    version=config.repo.version,
    openapi_tags=TAGS_METADATA,
    docs_url=None,
    redoc_url=None,
)

app.include_router(computer_vision.router)
app.include_router(datascience_results.router)
app.include_router(analyses.router)

logger.info(f'veritree datascience service live: {datetime.now()}')


@app.get("/")
async def landing():
    """
    Unauthenticated service banner.

    :return: a greeting naming the service and current time
    """
    logger.info('client reached: app.get route /')
    return {"response": f"Welcome to the veritree datascience service: {datetime.now()}"}


@app.get("/health")
async def health():
    """Unauthenticated liveness probe for the ECS target group."""
    return {"status": "ok", "service": config.repo.name, "version": config.repo.version}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve the veritree mark as the browser tab icon.

    Browsers request /favicon.ico unprompted for any page that does not declare an
    icon link -- without this the plain JSON routes log a 404 on every visit.
    """
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    """
    Swagger UI, re-declared to use the veritree favicon.

    FastAPI's built-in docs route hardcodes its own CDN favicon, so the default is
    disabled on the app and replaced here.

    :return: the Swagger UI HTML page
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="/favicon.ico",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc():
    """
    ReDoc, re-declared to use the veritree favicon.

    :return: the ReDoc HTML page
    """
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_favicon_url="/favicon.ico",
    )
