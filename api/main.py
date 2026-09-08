"""
veritree datascience service -- FastAPI application entrypoint.

Route groups:
  /computer_vision      inference over field photo evidence
  /datascience_results  read-only results from the analytics database
  /analyses             per-request LLM/model-backed analyses over a supplied payload
"""

import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse

from api.routers import analyses, computer_vision, datascience_results

from src.utils import context


config = context.config
logger = context.logger


FAVICON_PATH = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')


# Rendered above the route list in /docs. Markdown is supported, but Swagger styles
# table cells generously -- a table whose cells hold sentences renders several hundred
# pixels tall -- so prose is kept in lists and tables are reserved for short values.
API_DESCRIPTION = """
Computer vision over field photo evidence, and read access to the results the
veritree datascience pipelines produce.

---

### Authentication

Every route except `/` and `/health` requires a `Token` header:

```http
Token: <API_ENDPOINT_TOKEN>
```

Click **Authorize** above and paste the token once — Swagger then sends it with every
*Try it out* request on this page.

---

### Route groups

**`computer_vision`** &nbsp;·&nbsp; one endpoint, three model pipelines, selected by a
`service` field. Takes one image in S3.

**`datascience_results`** &nbsp;·&nbsp; results from the upstream pipelines, read either
from the analytics database or by scanning S3 directly (`live_scan=true`).

**`analyses`** &nbsp;·&nbsp; per-request LLM analyses over a payload you supply. No image
involved.

---

### Conventions

- **Paging** — `limit` and `offset` on every list route. `n_records` is the total
  matching your filters; `n_returned` is the size of this page.
- **Filters** — all optional. Omitting one widens the result rather than narrowing it.
- **Status codes** — `501` means a source is not available here (not implemented, or
  not yet published upstream) rather than a fault; `502` means an upstream store —
  S3 or the database — failed.
"""

TAGS_METADATA = [
    {
        "name": "computer_vision",
        "description": (
            "Model inference over a single field photo. One endpoint routes to three "
            "services via the **`service`** field:\n\n"
            "| `service` | Does |\n"
            "|---|---|\n"
            "| `survivability_detection` | mangrove detection + alive/dead/unclear |\n"
            "| `content_tagging` | L3 verification tags (`people`, `meterstick`, ...) |\n"
            "| `content_moderation` | flags images needing human review |\n\n"
            "**Addressing the image.** `image_url` takes a full `s3://bucket/key` URI, a "
            "bare object key resolved against the configured source bucket, or an "
            "`https://` URL fetched over HTTP.\n\n"
            "**Cost model.** The image is read and decoded once per request whichever "
            "service runs, and is never written back — a photo is stored exactly once. "
            "Model weights load at startup, not per request.\n\n"
            "**Chaining.** `content_tagging` routes into moderation automatically when it "
            "detects `people`, and merges those tags into its own."
        ),
    },
    {
        "name": "datascience_results",
        "description": (
            "Results the upstream pipelines produce. These routes compute nothing.\n\n"
            "| Route | Source |\n"
            "|---|---|\n"
            "| `bioacoustics_results` | analytics database |\n"
            "| `multispectral_results` | **`live_scan=true`** — S3 |\n"
            "| `treetracker_results` | **`live_scan=true`** — S3 |\n\n"
            "**Live scan.** Indexes the assets straight out of S3 instead of reading a "
            "published table, so a result appears as soon as the pipeline writes it. The "
            "response reports where rows came from in `source_type`. The database sources "
            "for the two S3-backed routes are not implemented, so they answer `501` "
            "without the flag.\n\n"
            "**Paging.** Database reads apply `limit`/`offset` in SQL alongside a count, "
            "so a response is bounded by the page rather than the table."
        ),
    },
    {
        "name": "analyses",
        "description": (
            "Per-request LLM analyses over a payload the caller supplies.\n\n"
            "Distinct from **computer_vision**, which needs an image, and from "
            "**datascience_results**, which serves rows a pipeline already produced."
        ),
    },
    {
        "name": "service",
        "description": (
            "Unauthenticated service endpoints. `/health` is what the ECS target group "
            "polls; its `timestamp` is an ISO-8601 instant carrying its UTC offset."
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

logger.info(f"api ready: service={config.repo.name} version={config.repo.version}")


@app.get("/", tags=["service"], summary="Service banner")
async def landing():
    """
    Unauthenticated service banner.

    :return: a greeting naming the service and current time
    """
    return {"response": f"Welcome to the veritree datascience service: {datetime.now()}"}


@app.get("/health", tags=["service"], summary="Liveness probe",
         response_description="Service identity, version and the current UTC timestamp")
async def health():
    """
    Unauthenticated liveness probe for the ECS target group.

    ``timestamp`` is an ISO-8601 instant carrying its UTC offset, so a reader can
    convert it to their own zone without knowing where the task ran. Seconds
    precision: a probe does not need milliseconds, and the shorter string stays
    readable in logs.

    :return: service identity, version, and the current timestamp
    """

    return {
        "status": "ok",
        "service": config.repo.name,
        "version": config.repo.version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
