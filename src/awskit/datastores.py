"""
Read-only accessors for the veritree analytics Postgres instance.

One section per domain, mirroring the /datascience_results route groups:

  bioacoustics   -> bioacoustics.tbl_postprocess_site_level_indicies
  multispectral  -> multispectral.tbl_raster_results
  treetracker    -> pending, see :func:`get_treetracker_results`

These live in separate databases on separate servers, so each accessor resolves its
own engine by domain rather than sharing one. A domain with no configured endpoint,
or whose table is absent, raises :class:`ResultsTableUnavailable` -- the route turns
that into a 501, which reads as "not available here" rather than a server fault.

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
from sqlalchemy.exc import ProgrammingError

from src.awskit.datahandlers import DatabaseNotConfigured, get_handler

from src.utils import context


config = context.config
logger = context.logger


# Identifiers are format-interpolated into SQL, so they must never come from request
# data. These allowlists pin them to values defined in this module.
BIOACOUSTICS_TABLE = 'bioacoustics.tbl_postprocess_site_level_indicies'
MULTISPECTRAL_TABLE = 'multispectral.tbl_raster_results'

# Domain keys, matching src.awskit.datahandlers.DOMAIN_ENV_PREFIXES
BIOACOUSTICS_DOMAIN = 'bioacoustics'
MULTISPECTRAL_DOMAIN = 'multispectral'
TREETRACKER_DOMAIN = 'treetracker'


class ResultsTableUnavailable(NotImplementedError):
    """
    The upstream pipeline has not published its results table to this database yet.

    Distinct from a query failure: nothing is broken here, the data simply is not
    there. The routes map it to 501 so a caller can tell "not published yet" apart
    from "the service is misconfigured".
    """


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


def _fetch_page(domain: str, table: str, filters: dict,
                limit: int, offset: int) -> ResultPage:
    """
    Fetch one page from a domain's table, plus the total matching the same filters.

    Two round trips rather than one: counting in SQL is what keeps the row fetch
    bounded, and both statements hit the same filtered index.

    :param domain: which database to query, e.g. 'bioacoustics'
    :param table: schema-qualified table name; must be a module constant, never
        request data, since it is interpolated rather than bound
    :param filters: mapping of column name to value; None values are ignored
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: the page and the pre-paging total
    :raises ResultsTableUnavailable: if the domain has no database configured, or the
        table is absent from it
    :raises Exception: any other failure, re-raised after logging
    """

    where, params = _build_where(filters)

    try:
        handler = get_handler(domain)
    except DatabaseNotConfigured as e:
        logger.warning(f"datastores domain unconfigured: domain={domain} error={e}")
        raise ResultsTableUnavailable(str(e)) from e

    try:
        # No-op unless tunnelling; restarts a dropped SSH forward before the pool
        # hands out a connection through a local port that no longer exists.
        handler.ensure_ready()

        with handler.engine.connect() as connection:
            total = connection.execute(
                text(f"select count(*) from {table}{where}"), params
            ).scalar_one()

            rows = pd.read_sql_query(
                text(f"select * from {table}{where} limit :_limit offset :_offset"),
                con=connection,
                params={**params, '_limit': limit, '_offset': offset},
            )

    except ProgrammingError as e:
        if 'UndefinedTable' in str(type(e.orig)) or 'does not exist' in str(e.orig):
            logger.warning(f"datastores table absent: table={table} "
                           f"reason=upstream_has_not_published")
            raise ResultsTableUnavailable(
                f"'{table}' has not been published to this database yet") from e

        logger.error(f"datastores query rejected: table={table} error={e}")
        raise

    except Exception as e:
        logger.error(f"datastores query failed: table={table} error={e}")
        raise

    # A surrogate index column from pandas' to_sql on the writing side; not data.
    if 'index' in rows.columns:
        rows = rows.drop(columns=['index'])

    logger.info(f"datastores query ok: table={table} returned={len(rows)} "
                f"matched={total} limit={limit} offset={offset}")

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
        BIOACOUSTICS_DOMAIN,
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
    Get indexed multispectral raster outputs from the published table.

    NOT IMPLEMENTED. The route answers from a live S3 scan instead, which indexes the
    rasters straight out of the bucket and needs no publish step. To restore the
    database source, delegate to :func:`_fetch_page` with ``MULTISPECTRAL_DOMAIN`` and
    ``MULTISPECTRAL_TABLE`` as the bioacoustics accessor does, and drop the raise --
    the table, its columns and its filters are recorded in
    configs/config_datascience_results.yaml.

    :param country: optional country filter
    :param site: optional site filter
    :param subsite: optional subsite filter
    :param period: optional period filter
    :param raster: optional raster product filter, e.g. PRODUCTIVITY or STRUCTURE
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: page of raster results
    :raises ResultsTableUnavailable: always, until the database source is implemented
    """

    raise ResultsTableUnavailable(
        "The database source for multispectral_results is not implemented; "
        "pass live_scan=true to index the rasters directly from S3")


def get_treetracker_results(country: Optional[str] = None,
                            site: Optional[str] = None,
                            session_id: Optional[str] = None,
                            limit: int = 1000,
                            offset: int = 0) -> ResultPage:
    """
    Get processed GPS tree-tracker session results from a published table.

    NOT IMPLEMENTED, and nothing upstream publishes one: veritree-tree-tracker-algorithms
    writes only the staging/fact tables it consumes plus dim_* rollups. The route
    answers from a live S3 scan of the session assets instead. To add a database
    source, name the table in configs/config_datascience_results.yaml, add a module
    constant for it, delegate to :func:`_fetch_page`, and drop the raise.

    :param country: optional country filter
    :param site: optional site filter
    :param session_id: optional planting session filter
    :param limit: maximum rows to return
    :param offset: rows to skip
    :return: page of tree-tracker session results
    :raises ResultsTableUnavailable: always, until a results table exists upstream
    """

    raise ResultsTableUnavailable(
        "The database source for treetracker_results is not implemented; "
        "pass live_scan=true to index the session assets directly from S3")
