"""
Read-only accessors for the veritree analytics Postgres instance.

One section per domain, mirroring the datascience_results router groups:

  bioacoustics   -> bioacoustics.tbl_postprocess_site_level_indicies
  multispectral  -> multispectral.tbl_raster_results
  treetracker    -> (pending: see get_treetracker_results)

Filters are bound as SQLAlchemy parameters rather than interpolated, because these
are reached from public API routes. Only filters the caller actually supplied are
added to the WHERE clause -- a `(:param is null or col = :param)` form would send
Postgres an untyped NULL and fail with "could not determine data type of parameter",
as well as defeating index use.
"""

import pandas as pd
from sqlalchemy import text

from src.awskit.datahandlers import pg_handler

from src.utils import context


config = context.config
logger = context.logger


def _build_where(filters: dict) -> tuple:
    """
    Build a WHERE clause from the filters that were actually supplied.

    :param filters: mapping of column name -> value (None values are skipped)
    :return: (where_clause_sql, params) where the clause is '' when nothing was supplied
    """

    active = {column: value for column, value in filters.items() if value is not None}

    if not active:
        return '', {}

    clause = " where " + " and ".join(f"{column} = :{column}" for column in active)

    return clause, active


def _read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Execute a read-only query and return a DataFrame, logging and re-raising failures
    so the router can translate them into a 5xx rather than silently returning empties.
    """

    try:
        df = pd.read_sql_query(text(sql), con=pg_handler.engine, params=params or {})
    except Exception as e:
        logger.error(f"Query failed: {' '.join(sql.split())[:120]}... due to: {e}")
        raise

    logger.info(f"Query returned {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Bioacoustics
# Source: veritree_bioacoustics :: task_postprocess_results
# ---------------------------------------------------------------------------

def get_postprocess_site_level_indicies(code_country: str | None = None,
                                        code_site: str | None = None,
                                        recording_year: str | None = None) -> pd.DataFrame:
    """
    Get postprocessed site-species-year biodiversity indices.

    :param code_country: optional country code filter
    :param code_site: optional site code filter
    :param recording_year: optional recording year filter
    :return: DataFrame of site level indices
    """

    where, params = _build_where({
        'code_country': code_country,
        'code_site': code_site,
        'recording_year': recording_year,
    })

    df = _read_sql(f"select * from bioacoustics.tbl_postprocess_site_level_indicies{where}", params)

    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    return df


# ---------------------------------------------------------------------------
# Multispectral forest health
# Source: veritree_multispectral_foresthealth :: task_publish_geospatial_service
# ---------------------------------------------------------------------------

def get_raster_results(country: str | None = None,
                       site: str | None = None,
                       subsite: str | None = None,
                       period: str | None = None,
                       raster: str | None = None) -> pd.DataFrame:
    """
    Get the indexed multispectral raster outputs joined to veritree planting site ids.

    Columns: project_name, country, site, region_id, subsite, site_id, period,
             capture_date, raster, raster_name, s3_key, s3_uri, https_url, generated_at

    :param country: optional country filter
    :param site: optional site filter
    :param subsite: optional subsite filter
    :param period: optional period filter
    :param raster: optional raster product filter (e.g. PRODUCTIVITY, STRUCTURE)
    :return: DataFrame of raster results
    """

    where, params = _build_where({
        'country': country,
        'site': site,
        'subsite': subsite,
        'period': period,
        'raster': raster,
    })

    df = _read_sql(f"select * from multispectral.tbl_raster_results{where}", params)

    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    return df


# ---------------------------------------------------------------------------
# Tree trackers
# Source: veritree-tree-tracker-algorithms
# ---------------------------------------------------------------------------

def get_treetracker_results(country: str | None = None,
                            site: str | None = None,
                            session_id: str | None = None) -> pd.DataFrame:
    """
    Get processed GPS tree-tracker session results.

    NOT YET IMPLEMENTED. veritree-tree-tracker-algorithms currently writes only the
    staging/fact tables it consumes (planting_sites, field_updates, allocations,
    tree_orders, capacities, planting_partners) plus dim_* rollups -- there is no
    published per-session results table equivalent to the bioacoustics and
    multispectral ones. Once that repo publishes one, set the table in
    configs/config_datascience_results.yaml, implement the query below in the same
    shape as the two above, and drop the NotImplementedError.

    :param country: optional country filter
    :param site: optional site filter
    :param session_id: optional planting session filter
    :return: DataFrame of tree-tracker session results
    """

    raise NotImplementedError(
        "treetracker results table is not yet published by veritree-tree-tracker-algorithms"
    )


# ---------------------------------------------------------------------------
# Shared reference tables
# ---------------------------------------------------------------------------

def get_planting_sites_table(schema: str = 'multispectral', cols: list | None = None) -> pd.DataFrame:
    """
    Get planting sites reference table
    """

    sub = ", ".join(cols) if cols else "*"

    return _read_sql(f"select {sub} from {schema}.tbl_planting_sites")


def get_field_updates_table(schema: str = 'multispectral', cols: list | None = None) -> pd.DataFrame:
    """
    Get field updates reference table
    """

    sub = ", ".join(cols) if cols else "*"

    return _read_sql(f"select {sub} from {schema}.tbl_field_updates")


def get_survivability_inferences() -> pd.DataFrame:
    """
    Get persisted survivability inference results
    """

    return _read_sql("select * from survivability.tbl_survivability_inference")
