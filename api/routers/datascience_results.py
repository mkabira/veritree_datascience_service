"""
Datascience results routes.

Read-only access to results already published to the analytics Postgres instance by
the upstream pipelines:

  /datascience_results/bioacoustics_results   bioacoustics.tbl_postprocess_site_level_indicies
  /datascience_results/multispectral_results  multispectral.tbl_raster_results
  /datascience_results/treetracker_results    (pending upstream results table)

These routes do not compute anything -- they serve what the pipelines wrote.
"""

import math
import time
import uuid
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import api_authentication

from src.awskit import datastores

from src.services import live_scan as live_scan_module

from src.utils import context


config = context.config
logger = context.logger


router = APIRouter(prefix="/datascience_results", tags=["datascience_results"])


MAX_PAGE_SIZE = 5000


class ResultsResponse(BaseModel):
    """One page of results, with the total matching the filters."""

    source_type: str = Field(
        'database',
        description="Where the rows came from: 'database' or 'live_scan' (S3)")

    source:      str
    n_records:   int   # total rows matching the filters
    n_returned:  int   # rows in this page, after limit/offset
    limit:       int
    offset:      int
    results:     list


def _to_records(df: pd.DataFrame) -> list:
    """
    Render a results frame JSON-safe.

    The frame is already the requested page: LIMIT and OFFSET are applied by the
    database, so no slicing happens here. NaN/NaT are not valid JSON and are replaced
    with None rather than emitted as the literal `NaN` pandas' own to_json produces;
    Timestamps become ISO-8601 strings.

    :param df: one page of query results
    :return: list of JSON-serialisable row dicts
    """

    records = df.to_dict(orient='records')

    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
            elif value is pd.NaT:
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()

    return records


def _live_scan_response(source: str, scan, limit: int, offset: int,
                        request_id: str) -> ResultsResponse:
    """
    Run a live S3 scan and page its rows into the shared response envelope.

    Unlike the database path, paging happens in Python: S3 has no equivalent of
    LIMIT/OFFSET, so the whole prefix is listed either way and slicing here costs
    nothing extra.

    :param source: human-readable origin, recorded in the response
    :param scan: zero-argument callable returning the scanned rows
    :param limit: maximum rows to return in this page
    :param offset: rows to skip before this page
    :param request_id: correlation id for logging
    :return: the page, with source_type 'live_scan'
    :raises HTTPException: 501 if the domain has no scan configured, 502 if S3 fails,
        504 if the scan exceeds its time budget
    """

    started = time.monotonic()

    try:
        rows = scan()
    except live_scan_module.LiveScanNotConfigured as e:
        logger.warning(f"live_scan unavailable: request_id={request_id} error={e}")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

    except live_scan_module.LiveScanTimeout as e:
        # 504 rather than 502: the upstream did not fail, we stopped waiting for it.
        logger.error(f"live_scan timed out: request_id={request_id} error={e}")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except Exception as e:
        logger.error(f"live_scan failed: request_id={request_id} error={e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Live scan of S3 failed; see the service logs for the cause")

    page = rows[offset:offset + limit]

    logger.info(f"live_scan finished: request_id={request_id} source={source} "
                f"matched={len(rows)} returned={len(page)} "
                f"duration={time.monotonic() - started:.2f}s")

    return ResultsResponse(
        source=source,
        source_type='live_scan',
        n_records=len(rows),
        n_returned=len(page),
        limit=limit,
        offset=offset,
        results=page)


@router.get(
    "/bioacoustics_results/",
    response_model=ResultsResponse,
    summary="Bioacoustics site-level biodiversity indices",
    response_description="Site-species-year rows from `bioacoustics.tbl_postprocess_site_level_indicies`",
    dependencies=[Depends(api_authentication)],
    responses={
        401: {"description": "Missing or invalid `Token` header"},
        422: {"description": "`limit` or `offset` outside the accepted range"},
        501: {"description": "Upstream has not published this table to the database yet"},
        500: {"description": "The query against the analytics database failed"},
    },
)
# Deliberately `def`, not `async def`: the body performs blocking I/O (database,
# S3, model inference or an HTTP call to a provider). FastAPI runs a sync handler in
# a threadpool, so the event loop stays free; the same body in an `async def` would
# stall every other request, including /health, for its whole duration.
def bioacoustics_results(
    code_country:   Optional[str] = Query(None, description="Country code filter"),
    code_site:      Optional[str] = Query(None, description="Site code filter"),
    recording_year: Optional[str] = Query(None, description="Recording year filter"),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Site-species-year biodiversity indices derived from BirdNET classification of
    field recordings, from bioacoustics.tbl_postprocess_site_level_indicies.

    :param code_country: optional country code filter
    :param code_site: optional site code filter
    :param recording_year: optional recording year filter
    :param limit: maximum rows to return in this page
    :param offset: rows to skip before this page
    :return: the page, plus the total matching the filters
    :raises HTTPException: 500 if the query fails
    """

    request_id = str(uuid.uuid4())

    logger.info(f"datascience_results bioacoustics_results started: request_id={request_id} "
                f"country={code_country} site={code_site} year={recording_year}")

    try:
        page = datastores.get_postprocess_site_level_indicies(
            code_country=code_country,
            code_site=code_site,
            recording_year=recording_year,
            limit=limit,
            offset=offset)

        records = _to_records(page.rows)

        logger.info(f"datascience_results bioacoustics_results finished: request_id={request_id} "
                    f"source=database matched={page.total} returned={len(records)}")

        return ResultsResponse(
            source='bioacoustics.tbl_postprocess_site_level_indicies',
            n_records=page.total,
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except NotImplementedError as e:
        logger.warning(f"datascience_results bioacoustics_results unavailable: request_id={request_id} error={e}")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

    except Exception as e:
        # The exception carries the failing SQL; log it, but do not return it. A
        # client cannot act on it and it describes the schema to anyone who asks.
        logger.error(f"datascience_results bioacoustics_results failed: request_id={request_id} error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch bioacoustics_results; see the service logs for the cause")


@router.get(
    "/multispectral_results/",
    response_model=ResultsResponse,
    summary="Multispectral forest-health raster index (live S3 scan)",
    response_description="Raster outputs from `multispectral.tbl_raster_results`, with their S3 and HTTPS locations",
    dependencies=[Depends(api_authentication)],
    responses={
        401: {"description": "Missing or invalid `Token` header"},
        422: {"description": "`limit` or `offset` outside the accepted range"},
        501: {"description": "The database source is not implemented; pass live_scan=true"},
        502: {"description": "The live S3 scan failed"},
        504: {"description": "The live S3 scan exceeded its time budget"},
    },
)
def multispectral_results(
    country: Optional[str] = Query(None, description="Country filter"),
    site:    Optional[str] = Query(None, description="Site filter"),
    subsite: Optional[str] = Query(None, description="Subsite filter"),
    period:  Optional[str] = Query(None, description="Period filter"),
    raster:  Optional[str] = Query(None, description="Raster product, e.g. PRODUCTIVITY"),
    live_scan: bool = Query(
        False,
        description="Index the assets directly from S3 instead of reading the "
                    "published table. Required for now: the database source is "
                    "not implemented."),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Multispectral forest-health raster outputs by site and period, with the S3 and
    HTTPS location of each cloud-optimized raster.

    **Pass `live_scan=true`.** The database source is not implemented, so the
    published-table path returns 501. A live scan indexes the rasters straight out of
    S3, deriving each row's country/site/subsite/period from the key layout, so a
    raster is visible as soon as the pipeline writes it.

    :param country: optional country filter
    :param site: optional site filter
    :param subsite: optional subsite filter
    :param period: optional period filter
    :param raster: optional raster product filter, e.g. PRODUCTIVITY or STRUCTURE
    :param live_scan: index from S3 rather than the published table
    :param limit: maximum rows to return in this page
    :param offset: rows to skip before this page
    :return: the page, plus the total matching the filters
    :raises HTTPException: 501 without live_scan, 502 if the S3 scan fails
    """

    request_id = str(uuid.uuid4())

    logger.info(f"datascience_results multispectral_results started: request_id={request_id} "
                f"live_scan={live_scan} country={country} site={site} raster={raster}")

    if live_scan:
        return _live_scan_response(
            source=f"s3://{config.datascience_results.multispectral.live_scan.bucket}",
            scan=lambda: live_scan_module.scan_multispectral_rasters(
                country=country, site=site, subsite=subsite,
                period=period, raster=raster),
            limit=limit, offset=offset, request_id=request_id)

    # Not implemented today: the accessor raises ResultsTableUnavailable, which the
    # handler below turns into a 501. Keeping the call here rather than short-
    # circuiting means implementing the database source needs no change to this route.
    try:
        page = datastores.get_raster_results(
            country=country, site=site, subsite=subsite, period=period, raster=raster,
            limit=limit,
            offset=offset)

        records = _to_records(page.rows)

        logger.info(f"datascience_results multispectral_results finished: request_id={request_id} "
                    f"source=database matched={page.total} returned={len(records)}")

        return ResultsResponse(
            source=f"{datastores.MULTISPECTRAL_TABLE}",
            n_records=page.total,
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except NotImplementedError as e:
        logger.warning(f"datascience_results multispectral_results unavailable: request_id={request_id} error={e}")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

    except Exception as e:
        logger.error(f"datascience_results multispectral_results failed: request_id={request_id} error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch multispectral_results; see the service logs for the cause")


@router.get(
    "/treetracker_results/",
    response_model=ResultsResponse,
    summary="GPS tree-tracker session assets (live S3 scan)",
    response_description="Returns 501 until the upstream pipeline publishes a results table",
    dependencies=[Depends(api_authentication)],
    responses={
        401: {"description": "Missing or invalid `Token` header"},
        422: {"description": "`limit` or `offset` outside the accepted range"},
        501: {"description": "The database source is not implemented; pass live_scan=true"},
        502: {"description": "The live S3 scan failed"},
        504: {"description": "The live S3 scan exceeded its time budget"},
    },
)
def treetracker_results(
    country:    Optional[str] = Query(None, description="Country filter"),
    site:       Optional[str] = Query(None, description="Site filter"),
    session_id: Optional[str] = Query(None, description="Planting session filter"),
    live_scan: bool = Query(
        False,
        description="Index the assets directly from S3 instead of reading the "
                    "published table. Required for now: the database source is "
                    "not implemented."),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Per-session GPS tree-tracker results.

    **Pass `live_scan=true`.** veritree-tree-tracker-algorithms publishes no results
    table -- only staging/fact tables and dim_* rollups -- so the database path
    returns 501. A live scan indexes the per-session GeoJSON tracks straight out of
    S3, deriving the session, device and capture date from each key.

    `country` and `site` are accepted for parity with the other result routes but are
    not encoded in S3 keys, so they are ignored during a live scan; filter on
    `session_id` instead.

    :param country: ignored during a live scan, see above
    :param site: ignored during a live scan, see above
    :param session_id: optional planting session filter
    :param live_scan: index from S3 rather than the published table
    :param limit: maximum rows to return in this page
    :param offset: rows to skip before this page
    :return: the page, plus the total matching the filters
    :raises HTTPException: 501 without live_scan, 502 if the S3 scan fails
    """

    request_id = str(uuid.uuid4())

    logger.info(f"datascience_results treetracker_results started: request_id={request_id} "
                f"live_scan={live_scan} planting_session={session_id}")

    if live_scan:
        return _live_scan_response(
            source=f"s3://{config.datascience_results.treetracker.live_scan.bucket}",
            scan=lambda: live_scan_module.scan_treetracker_assets(
                country=country, site=site, session_id=session_id),
            limit=limit, offset=offset, request_id=request_id)

    # Not implemented today: the accessor raises ResultsTableUnavailable, which the
    # handler below turns into a 501. Keeping the call here rather than short-
    # circuiting means implementing the database source needs no change to this route.
    try:
        page = datastores.get_treetracker_results(
            country=country, site=site, session_id=session_id,
            limit=limit,
            offset=offset)

        records = _to_records(page.rows)

        logger.info(f"datascience_results treetracker_results finished: request_id={request_id} "
                    f"source=database matched={page.total} returned={len(records)}")

        return ResultsResponse(
            source=f"{datastores.TREETRACKER_DOMAIN}",
            n_records=page.total,
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except NotImplementedError as e:
        logger.warning(f"datascience_results treetracker_results unavailable: request_id={request_id} error={e}")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

    except Exception as e:
        logger.error(f"datascience_results treetracker_results failed: request_id={request_id} error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch treetracker_results; see the service logs for the cause")
