"""
Read-only accessors for the veritree analytics Postgres instance.

One section per domain, mirroring the /datascience_results route groups:

  bioacoustics   -> bioacoustics.tbl_postprocess_site_level_indicies
  multispectral  -> multispectral.tbl_raster_results
  treetracker    -> pending, see :func:`get_treetracker_results`

Every accessor returns a :class:`ResultPage`: the requested slice of rows plus the
total number matching the filters. Both LIMIT/OFFSET and the count are evaluated by
the database, so response size is bounded by the page rather than by table size --
these tables grow with every survey season and a `select *` would eventually
materialise the whole thing into a DataFrame on each request.

Filters are bound as SQLAlchemy parameters rather than interpolated, because they
arrive from public API routes. Only filters the caller supplied enter the WHERE
clause: a `(:param is null or col = :param)` form would send Postgres an untyped
NULL and fail with "could not determine data type of parameter", as well as
defeating index use.
"""

from typing import NamedTuple, Optional

import pandas as pd
from sqlalchemy import text

from src.awskit.datahandlers import pg_handler

from src.utils import context


config = context.config
logger = context.logger


# Identifiers are format-interpolated into SQL, so they must never come from request
# data. These allowlists pin them to values defined in this module.
BIOACOUSTICS_TABLE = 'bioacoustics.tbl_postprocess_site_level_indicies'
MULTISPECTRAL_TABLE = 'multispectral.tbl_raster_results'


class ResultPage(NamedTuple):
    """
    One page of query results.

    :param rows: the requested slice, already limited and offset by the database
    :param total: rows matching the filters before paging, for the caller's metadata
    """

    rows: pd.DataFrame
    total: int


def _build_where(filters: dict) -> tuple:
    """
    Build a WHERE clause from the filters that were actually supplied.

    :param filters: mapping of column name to value; None values are skipped
    :return: (clause, params) where clause is '' when no filter was supplied
    """

    active = {column: value for column, value in filters.items() if value is not None}

    if not active:
        return '', {}

    clause = " where " + " and ".join(f"{column} = :{column}" for column in active)

    return clause, active


def _fetch_page(table: str, filters: dict, limit: int, offset: int) -> ResultPage:
    """
    Fetch one page from a table, plus the total matching the same filters.

    Two round trips rather than one: counting in SQL is what keeps the row fetch
    bounded, and both statements hit the same filtered index.

    :param table: schema-qualified table name; must be a module constant, never
        request data, since it is interpolated rather than bound
    :param filters: mapping of column name to value; None values are ignored
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: the page and the pre-paging total
    :raises Exception: re-raised after logging so the route maps it to a 5xx
    """

    where, params = _build_where(filters)

    try:
        with pg_handler.engine.connect() as connection:
            total = connection.execute(
                text(f"select count(*) from {table}{where}"), params
            ).scalar_one()

            rows = pd.read_sql_query(
                text(f"select * from {table}{where} limit :_limit offset :_offset"),
                con=connection,
                params={**params, '_limit': limit, '_offset': offset},
            )

    except Exception as e:
        logger.error(f"Query against {table} failed: {e}")
        raise

    # A surrogate index column from pandas' to_sql on the writing side; not data.
    if 'index' in rows.columns:
        rows = rows.drop(columns=['index'])

    logger.info(f"{table}: returned {len(rows)} of {total} matching rows "
                f"(limit={limit} offset={offset})")

    return ResultPage(rows=rows, total=total)


def get_postprocess_site_level_indicies(code_country: Optional[str] = None,
                                        code_site: Optional[str] = None,
                                        recording_year: Optional[str] = None,
                                        limit: int = 1000,
                                        offset: int = 0) -> ResultPage:
    """
    Get postprocessed site-species-year biodiversity indices.

    Written by veritree_bioacoustics :: task_postprocess_results.

    :param code_country: optional country code filter
    :param code_site: optional site code filter
    :param recording_year: optional recording year filter
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: page of site-level indices and the pre-paging total
    """

    return _fetch_page(
        BIOACOUSTICS_TABLE,
        {'code_country': code_country, 'code_site': code_site,
         'recording_year': recording_year},
        limit, offset)


def get_raster_results(country: Optional[str] = None,
                       site: Optional[str] = None,
                       subsite: Optional[str] = None,
                       period: Optional[str] = None,
                       raster: Optional[str] = None,
                       limit: int = 1000,
                       offset: int = 0) -> ResultPage:
    """
    Get indexed multispectral raster outputs joined to veritree planting site ids.

    Written by veritree_multispectral_foresthealth ::
    task_publish_geospatial_service. Columns: project_name, country, site, region_id,
    subsite, site_id, period, capture_date, raster, raster_name, s3_key, s3_uri,
    https_url, generated_at.

    :param country: optional country filter
    :param site: optional site filter
    :param subsite: optional subsite filter
    :param period: optional period filter
    :param raster: optional raster product filter, e.g. PRODUCTIVITY or STRUCTURE
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: page of raster results and the pre-paging total
    """

    return _fetch_page(
        MULTISPECTRAL_TABLE,
        {'country': country, 'site': site, 'subsite': subsite,
         'period': period, 'raster': raster},
        limit, offset)


def get_treetracker_results(country: Optional[str] = None,
                            site: Optional[str] = None,
                            session_id: Optional[str] = None,
                            limit: int = 1000,
                            offset: int = 0) -> ResultPage:
    """
    Get processed GPS tree-tracker session results.

    NOT YET IMPLEMENTED. veritree-tree-tracker-algorithms currently writes only the
    staging/fact tables it consumes (planting_sites, field_updates, allocations,
    tree_orders, capacities, planting_partners) plus dim_* rollups; there is no
    published per-session results table equivalent to the two above. Once that repo
    publishes one, set the table in configs/config_datascience_results.yaml, add a
    module constant for it, delegate to :func:`_fetch_page` as the others do, and
    drop the raise.

    :param country: optional country filter
    :param site: optional site filter
    :param session_id: optional planting session filter
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: page of tree-tracker session results
    :raises NotImplementedError: until the upstream results table exists
    """

    raise NotImplementedError(
        "treetracker results table is not yet published by veritree-tree-tracker-algorithms"
    )
