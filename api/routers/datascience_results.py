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
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.dependencies import api_authentication

from src.awskit import datastores

from src.utils import context


config = context.config
logger = context.logger


router = APIRouter(prefix="/datascience_results", tags=["datascience_results"])


MAX_PAGE_SIZE = 5000


class ResultsResponse(BaseModel):
    source:      str
    n_records:   int   # total rows matching the filters
    n_returned:  int   # rows in this page, after limit/offset
    limit:       int
    offset:      int
    results:     list


def _to_records(df: pd.DataFrame, limit: int, offset: int) -> list:
    """
    Slice a results frame and render it JSON-safe.

    NaN/NaT are not valid JSON, so they are replaced with None rather than being
    emitted as the literal `NaN` that pandas' own to_json would produce.
    """

    df_page = df.iloc[offset:offset + limit]

    records = df_page.to_dict(orient='records')

    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
            elif value is pd.NaT:
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()

    return records


@router.get("/bioacoustics_results/", response_model=ResultsResponse,
            dependencies=[Depends(api_authentication)])
async def bioacoustics_results(
    code_country:   Optional[str] = Query(None, description="Country code filter"),
    code_site:      Optional[str] = Query(None, description="Site code filter"),
    recording_year: Optional[str] = Query(None, description="Recording year filter"),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Site-species-year biodiversity indices derived from BirdNET classification of
    field recordings, from bioacoustics.tbl_postprocess_site_level_indicies.
    """

    logger.info(f"datascience_results bioacoustics_results: country={code_country} site={code_site}")

    try:
        df = datastores.get_postprocess_site_level_indicies(
            code_country=code_country,
            code_site=code_site,
            recording_year=recording_year)

        records = _to_records(df, limit, offset)

        return ResultsResponse(
            source='bioacoustics.tbl_postprocess_site_level_indicies',
            n_records=len(df),
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except Exception as e:
        err_string = f"Exception while fetching bioacoustics_results: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)


@router.get("/multispectral_results/", response_model=ResultsResponse,
            dependencies=[Depends(api_authentication)])
async def multispectral_results(
    country: Optional[str] = Query(None, description="Country filter"),
    site:    Optional[str] = Query(None, description="Site filter"),
    subsite: Optional[str] = Query(None, description="Subsite filter"),
    period:  Optional[str] = Query(None, description="Period filter"),
    raster:  Optional[str] = Query(None, description="Raster product, e.g. PRODUCTIVITY"),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Multispectral forest-health raster outputs indexed by site and period, with the
    S3 and HTTPS locations of each cloud-optimized raster, from
    multispectral.tbl_raster_results.
    """

    logger.info(f"datascience_results multispectral_results: country={country} site={site} raster={raster}")

    try:
        df = datastores.get_raster_results(
            country=country,
            site=site,
            subsite=subsite,
            period=period,
            raster=raster)

        records = _to_records(df, limit, offset)

        return ResultsResponse(
            source='multispectral.tbl_raster_results',
            n_records=len(df),
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except Exception as e:
        err_string = f"Exception while fetching multispectral_results: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)


@router.get("/treetracker_results/", response_model=ResultsResponse,
            dependencies=[Depends(api_authentication)])
async def treetracker_results(
    country:    Optional[str] = Query(None, description="Country filter"),
    site:       Optional[str] = Query(None, description="Site filter"),
    session_id: Optional[str] = Query(None, description="Planting session filter"),
    limit:  int = Query(1000, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """
    Per-session GPS tree-tracker results.

    Returns 501 until veritree-tree-tracker-algorithms publishes a results table --
    it currently writes only staging/fact tables and dim_* rollups. See
    src.awskit.datastores.get_treetracker_results.
    """

    logger.info(f"datascience_results treetracker_results: country={country} site={site}")

    try:
        df = datastores.get_treetracker_results(
            country=country,
            site=site,
            session_id=session_id)

        records = _to_records(df, limit, offset)

        return ResultsResponse(
            source='treetracker.tbl_treetracker_results',
            n_records=len(df),
            n_returned=len(records),
            limit=limit,
            offset=offset,
            results=records)

    except NotImplementedError as e:
        logger.warning(f"treetracker_results not yet available: {e}")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

    except Exception as e:
        err_string = f"Exception while fetching treetracker_results: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)
