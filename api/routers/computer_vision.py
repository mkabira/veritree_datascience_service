"""
Computer vision route.

One endpoint, ``POST /computer_vision/``, dispatching to a service named in the
request body:

  survivability_detection   mangrove detection + alive/dead/unclear classification
  content_tagging           field photo evidence tagging (people, meterstick, ...)
  content_moderation        AI content moderation over field photos

All three take the same input -- one image in the veritree S3 bucket -- so the
source is read and decoded ONCE here and the decoded image handed to whichever
service was asked for. Each returns its own ``result`` shape inside a common
envelope.

Ported from veritree-survivability :: api/surv_endpoint.py, which exposed these as
six separate routes. Anthropic is the only tagging backend: it measured best on
veritree field photos, so the OpenCLIP, Gemini and OpenAI taggers were retired.
"""

import io
import os
import time
import uuid
from functools import lru_cache
from typing import Annotated, List, Literal, Optional, Union

from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import api_authentication

from src.awskit.datahandlers import S3StorageHandler

from src.libs.veritag import load_photo

from src.services.computer_vision.survivability_detection import (
    detect_survivability,
    load_models as load_survivability_models,
)
from src.services.computer_vision.content_tagging import run_veritag_anthropic

from src.utils import context


config = context.config
logger = context.logger


router = APIRouter(prefix="/computer_vision", tags=["computer_vision"])


# Only a source bucket is needed: every service reads an image and returns JSON, so
# nothing is written back to S3.
#
# Two env-var shapes are accepted. AWS_SOURCE_* is what veritree-survivability used and
# what the ECS task definitions set; AWS_S3_* is the shape every other veritree
# datascience repo uses, so a .env copied from one of those works unchanged. The
# AWS_SOURCE_* names win when both are present.
DEFAULT_BUCKET_ENV = {
    "aws_accesskey": ("AWS_SOURCE_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
    "aws_secretkey": ("AWS_SOURCE_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
    "aws_regionname": ("AWS_SOURCE_REGION_NAME", "AWS_S3_REGION_NAME"),
    "aws_bucketname": ("AWS_SOURCE_S3_BUCKET_NAME", "AWS_S3_BUCKET_NAME"),
}


def _bucket_settings() -> dict:
    """
    Resolve the default bucket's credentials from whichever env-var shape is set.

    :return: kwargs for :class:`S3StorageHandler`
    """

    return {key: next((os.getenv(name) for name in names if os.getenv(name)), None)
            for key, names in DEFAULT_BUCKET_ENV.items()}


s3_source_handler = S3StorageHandler(prod_mode=config.prod_mode, **_bucket_settings())

if not s3_source_handler.connected():
    logger.warning(
        "computer_vision no default bucket: set AWS_SOURCE_S3_BUCKET_NAME (or "
        "AWS_S3_BUCKET_NAME) to read images by bare object key; requests using a full "
        "s3://bucket/key URI name their own bucket and still work")


@lru_cache(maxsize=8)
def _bucket_handler(bucket_name: str) -> S3StorageHandler:
    """
    Handler for a bucket named explicitly in an ``s3://`` URI.

    Cached: building a boto3 session per request would add a needless handshake, and
    callers realistically use a small, fixed set of buckets.

    :param bucket_name: bucket parsed out of the request's image_url
    :return: a read-only handler for that bucket
    """

    settings = _bucket_settings()
    settings["aws_bucketname"] = bucket_name

    logger.info(f"computer_vision opening bucket from uri: bucket={bucket_name}")

    return S3StorageHandler(prod_mode=config.prod_mode, **settings)


def _resolve_source(image_url: str):
    """
    Work out which bucket and key an image_url refers to.

    Three addressing forms are accepted, because all three are in use across the
    veritree services:

    * ``s3://bucket/key`` -- names its own bucket, so it works without a configured
      default
    * ``key/path.jpg`` -- a bare object key, read from the default source bucket
    * ``https://...`` -- fetched over HTTP; handled by the caller, not here

    :param image_url: the request's image_url
    :return: (handler, key) for the object
    """

    if image_url.startswith('s3://'):
        bucket_name, _, key = image_url[len('s3://'):].partition('/')

        if not bucket_name or not key:
            raise ValueError(f"Malformed S3 URI '{image_url}'; expected s3://bucket/key")

        return _bucket_handler(bucket_name), key

    return s3_source_handler, image_url

# Pay the model-loading cost at import rather than on the first request.
load_survivability_models()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ComputerVisionInput(BaseModel):
    """Request body: which service to run, and the image to run it on."""

    service: Literal['survivability_detection', 'content_tagging', 'content_moderation'] = Field(
        ..., description="Which computer vision service to route the image to")
    image_url: str = Field(
        ...,
        description=(
            "The source image, addressed any of three ways: a full `s3://bucket/key` "
            "URI (names its own bucket), a bare object key resolved against the "
            "configured source bucket, or an `https://` URL fetched over HTTP."
        ),
        examples=["s3://veritree-impactteam-survivability/survivability/100/img.jpeg"])


REQUEST_EXAMPLES = {
    "survivability_detection": {
        "summary": "survivability_detection — count alive/dead mangroves",
        "description": (
            "Detects mangroves and classifies each as alive, dead or unclear. Returns "
            "`counts` plus a `detections` list, each carrying `bbox_xyxy` in absolute "
            "pixels of the source image and `bbox_xywhn` normalised."
        ),
        "value": {
            "service": "survivability_detection",
            "image_url": "bulk_uploads/33/2026-03-16/mangroves.jpg",
        },
    },
    "content_tagging": {
        "summary": "content_tagging — tag photo evidence for L3 verification",
        "description": (
            "Tags the image against the labels in `config_cv_content_tagging.yaml` "
            "(`people`, `meterstick`, ...). If `people` is detected the image is routed "
            "through content moderation automatically and those tags are merged in, so "
            "this can return moderation tags too."
        ),
        "value": {
            "service": "content_tagging",
            "image_url": "bulk_uploads/33/2026-03-16/planting_evidence.jpg",
        },
    },
    "content_moderation": {
        "summary": "content_moderation — flag images needing human review",
        "description": (
            "Runs AI Content Moderation over the image. Returns `flagged` plus the tags "
            "behind it, and `scores` for every candidate regardless of the configured "
            "threshold, so an alternative operating point can be assessed without "
            "redeploying."
        ),
        "value": {
            "service": "content_moderation",
            "image_url": "bulk_uploads/33/2026-03-16/planting_evidence.jpg",
        },
    },
    "https_url": {
        "summary": "any service — addressing the image by https:// URL",
        "description": (
            "`image_url` also accepts a full URL to the same object, fetched over HTTP "
            "instead of through the bucket handler."
        ),
        "value": {
            "service": "content_tagging",
            "image_url": "https://veritreephotos.s3.us-east-2.amazonaws.com/bulk_uploads/33/img.jpeg",
        },
    },
}


class BoundingBox(BaseModel):
    """One detected mangrove: where it is, and how it was classified."""

    detection_id: str = Field(..., description="Unique per detection, per request")
    index: int = Field(..., description="Position within this image's detections")
    status: Literal['alive', 'dead', 'unclear'] = Field(
        ..., description="Highest-probability class from the classifier")
    detection_confidence: float = Field(
        ..., description="Detector's confidence that this is a mangrove at all")
    bbox_xyxy: List[float] = Field(..., description="Absolute pixels [x1, y1, x2, y2]")
    bbox_xywhn: List[float] = Field(..., description="Normalised [x_center, y_center, w, h]")
    probabilities: dict


class SurvivabilityCounts(BaseModel):
    """Aggregate survival counts across all detections in one image."""

    number_mangroves: int = Field(..., description="Total mangroves detected")
    number_alive: int
    number_dead: int
    number_unclear: int = Field(
        ..., description="Detected, but the model could not determine survival status")


class SurvivabilityResult(BaseModel):
    """Result of ``survivability_detection``: counts plus per-mangrove boxes."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "counts": {"number_mangroves": 107, "number_alive": 78,
                   "number_dead": 4, "number_unclear": 25},
        "detections": [{
            "detection_id": "4cd88ae5-ab26-4772-9db0-5a75557051ce",
            "index": 0,
            "status": "alive",
            "detection_confidence": 0.8801,
            "bbox_xyxy": [854.39, 650.0, 904.03, 807.26],
            "bbox_xywhn": [0.457922, 0.674658, 0.025858, 0.145614],
            "probabilities": {"alive": 0.9999, "dead": 0.0, "unclear": 0.0001},
        }],
    }})

    counts: SurvivabilityCounts
    detections: List[BoundingBox]


class TaggingResult(BaseModel):
    """Result of ``content_tagging``: tags that cleared their threshold."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "tags": ["people", "meterstick"],
        "scores": {"people": 0.97, "meterstick": 0.88},
    }})

    tags: List[str]
    scores: Optional[dict] = Field(
        None, description="Every candidate tag's score, including those below threshold")


class ModerationResult(BaseModel):
    """Result of ``content_moderation``: whether the image needs human review."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "flagged": True,
        "tags": ["minor_flagged"],
        "scores": {"minor_flagged": 0.81},
    }})

    flagged: bool = Field(..., description="True when the image needs human review")
    tags: List[str]
    scores: Optional[dict] = Field(
        None, description="Every candidate tag's score, including those below threshold")


class ImageDimensions(BaseModel):
    """Source image size, the frame ``bbox_xyxy`` pixels are measured in."""

    width: int
    height: int


class ComputerVisionOutput(BaseModel):
    """
    Response envelope shared by every service.

    ``service`` identifies which ``result`` shape is present, so a client can branch
    on the field it sent rather than probing the payload.
    """

    service: str = Field(..., description="The service that produced `result`")
    session_id: str = Field(..., description="Correlation id, also written to the logs")
    image_url: str = Field(..., description="The source as supplied, echoed back")
    image: ImageDimensions = Field(
        ..., description="Source dimensions; the frame `bbox_xyxy` is measured in")
    result: Union[SurvivabilityResult, TaggingResult, ModerationResult] = Field(
        ..., description="Service-specific payload; branch on `service`")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(image_url: str, session_id: str) -> Image.Image:
    """
    Load the source image once, whichever way the caller addressed it.

    Accepts every form in use across the veritree services: a full ``s3://bucket/key``
    URI, a bare object key read from the default source bucket, or an ``https://`` URL
    fetched over HTTP. Reading here rather than inside each service means one fetch per
    request no matter which service runs, and no image ever touches local disk.

    :param image_url: an s3:// URI, a bare S3 object key, or an https:// URL
    :param session_id: correlation id, so a load failure ties to its request
    :return: the decoded PIL image
    :raises HTTPException: 502 if the source cannot be read, 400 if it is not an image
    """

    try:
        if image_url.startswith(('http://', 'https://')):
            return load_photo(image_url)

        handler, key = _resolve_source(image_url)

        return Image.open(io.BytesIO(handler.read_bytes(key)))

    except ValueError as e:
        # A malformed s3:// URI is the caller's mistake, not an upstream failure.
        err_string = f"Invalid image_url: {e}"
        logger.error(f"computer_vision image load rejected: session_id={session_id} "
                     f"image={image_url} error={e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_string)

    except UnidentifiedImageError as e:
        err_string = f"Source at {image_url} is not a readable image: {e}"
        logger.error(f"computer_vision image not decodable: session_id={session_id} "
                     f"image={image_url} error={e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_string)

    except Exception as e:
        err_string = f"Unable to read image {image_url}: {e}"
        logger.error(f"computer_vision image unreadable: session_id={session_id} "
                     f"image={image_url} error={e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=err_string)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

def _run_survivability_detection(image, session_id: str) -> SurvivabilityResult:
    """
    Detect mangroves and classify each as alive, dead or unclear.

    :param image: decoded PIL image
    :param session_id: correlation id for logging
    :return: counts and per-mangrove bounding boxes
    """

    results = detect_survivability(image, session_id=session_id)

    return SurvivabilityResult(counts=results['counts'], detections=results['detections'])


def _run_content_tagging(image, session_id: str) -> TaggingResult:
    """
    Tag an image against the L3 verification labels.

    When ``people`` is detected the image is additionally routed through content
    moderation and the moderation tags are merged in, preserving the chaining
    behaviour of the deployed veritag_anthropic route.

    :param image: decoded PIL image
    :param session_id: correlation id for logging
    :return: matched tags and every tag's score
    """

    cfg = config.content_tagging_anthropic

    matches, scores = run_veritag_anthropic(
        model_name=cfg.model_name,
        image=image,
        tag_names=cfg.verification_tags,
        system_prompt=cfg.system_prompt,
        threshold=cfg.verification_threshold,
        session_id=session_id)

    if 'people' in matches:
        logger.info(f"content_tagging chaining to moderation: session_id={session_id} reason=people_detected")

        moderation = _run_content_moderation(image, session_id)

        matches = list(matches) + list(moderation.tags)
        scores = dict(scores) | dict(moderation.scores or {})

        logger.warning(f"content_moderation flagged via chain: session_id={session_id} tags={moderation.tags} scores={moderation.scores}")
    else:
        logger.info(f"content_tagging finished: session_id={session_id} moderation=skipped")

    return TaggingResult(tags=list(matches), scores=dict(scores))


def _run_content_moderation(image, session_id: str) -> ModerationResult:
    """
    Run AI content moderation (AICM) over a field photo.

    A triage layer, never the final decision: a flagged image goes to a trained
    reviewer. The AICM prompt and its calibrated threshold are tuned against this
    model.

    :param image: decoded PIL image
    :param session_id: correlation id for logging
    :return: whether the image is flagged, and the moderation tags behind it
    """

    cfg = config.aicm_anthropic

    matches, scores = run_veritag_anthropic(
        model_name=cfg.model_name,
        image=image,
        tag_names=cfg.verification_tags,
        system_prompt=cfg.system_prompt,
        threshold=cfg.verification_threshold,
        session_id=session_id)

    flagged = bool(matches)

    if flagged:
        logger.warning(f"content_moderation flagged: session_id={session_id} tags={matches} scores={scores}")

    return ModerationResult(flagged=flagged, tags=list(matches), scores=dict(scores))


SERVICES = {
    'survivability_detection': _run_survivability_detection,
    'content_tagging': _run_content_tagging,
    'content_moderation': _run_content_moderation,
}


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ComputerVisionOutput,
    summary="Run a computer vision service on an image",
    response_description="The service's result, inside the shared envelope",
    dependencies=[Depends(api_authentication)],
    responses={
        400: {"description": "The source was fetched but is not a readable image"},
        401: {"description": "Missing or invalid `Token` header"},
        422: {"description": "Unknown `service`, or a missing required field"},
        502: {"description": "The source image could not be read from S3 or over HTTP"},
    },
)
# Deliberately `def`, not `async def`: the body performs blocking I/O (database,
# S3, model inference or an HTTP call to a provider). FastAPI runs a sync handler in
# a threadpool, so the event loop stays free; the same body in an `async def` would
# stall every other request, including /health, for its whole duration.
def computer_vision(
    params: Annotated[ComputerVisionInput, Body(openapi_examples=REQUEST_EXAMPLES)],
):
    """
    Run one computer vision service against one image.

    Pick the pipeline with **`service`**:

    | `service` | Returns |
    |---|---|
    | `survivability_detection` | `counts` (mangroves, alive, dead, unclear) and `detections` with bounding boxes |
    | `content_tagging` | `tags` that cleared their threshold, and `scores` for every candidate |
    | `content_moderation` | `flagged`, the moderation `tags`, and `scores` |

    **Addressing the image.** `image_url` takes a bare S3 object key
    (`bulk_uploads/33/2026-03-16/img.jpg`), read through the source bucket, or an
    `https://` URL to the same object, fetched over HTTP. The image is read and
    decoded once per request whichever service runs, and is never written back to S3
    or to local disk — a photo is stored exactly once.

    **The response envelope** is the same for every service: `service`, `session_id`,
    `image_url`, `image` (the pixel dimensions `bbox_xyxy` is measured in), and
    `result`. Branch on `service` to know which `result` shape you have.

    **Chaining.** `content_tagging` routes into content moderation automatically when
    it detects `people`, and merges the moderation tags into its own — so a tagging
    call can legitimately return moderation tags.

    :param params: {'service': ..., 'image_url': ...}
    :return: the envelope, carrying the service's own result shape
    :raises HTTPException: 400 if the source is not an image, 502 if it cannot be
        read, 500 if the service itself fails
    """

    session_id = str(uuid.uuid4())
    started = time.monotonic()

    logger.info(f"computer_vision {params.service} started: "
                f"session_id={session_id} image={params.image_url}")

    image = _load_image(params.image_url, session_id)

    try:
        with image:
            width, height = image.size
            result = SERVICES[params.service](image, session_id)

        logger.info(f"computer_vision {params.service} finished: session_id={session_id} duration={time.monotonic() - started:.2f}s")

        return ComputerVisionOutput(
            service=params.service,
            session_id=session_id,
            image_url=params.image_url,
            image=ImageDimensions(width=width, height=height),
            result=result)

    except Exception as e:
        err_string = f"Exception while processing computer_vision {params.service}: {e}"
        logger.error(f"computer_vision {params.service} failed: session_id={session_id} "
                     f"duration={time.monotonic() - started:.2f}s error={e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)
