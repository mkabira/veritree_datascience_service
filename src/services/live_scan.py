"""
Live S3 scans, as an alternative to reading published results out of the database.

Two result domains publish their outputs as objects in S3 before -- or instead of --
indexing them into Postgres. Scanning the bucket answers from the assets themselves,
so a result is available as soon as the pipeline writes it, with no publish step in
between.

  multispectral  cloud-optimized rasters, keyed
                 <base>/<country>/<site>/<subsite>/<period>/outputs/rasters/<file>.tif
  treetracker    per-session GeoJSON tracks, keyed
                 <prefix>/<planting_session_id>/<DEVICE>_<YYYYMMDD>_<HHMMSS>.geojson

Ported from the indexing tasks that build the published tables:
  veritree_multispectral_foresthealth :: task_publish_geospatial_service/index_geospatial_rasters.py
  veritree-tree-tracker-algorithms    :: src/payloads_transfers/index_assets.py

Those tasks write a CSV and a database table. These functions return rows, because
the caller is an API route rather than a batch job -- nothing is written anywhere.
"""

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.awskit.datahandlers import S3_CLIENT_CONFIG

from src.utils import context


config = context.config
logger = context.logger


# Properties lifted out of each tree-tracker track file's header.
ASSET_PROPERTIES = [
    "gps_point_count", "gps_point_count_preclip", "points_dropped",
    "clipped_max_distance_m", "coord_precision", "source_file_count",
]


# A scan lists a whole prefix, so its cost grows with the bucket rather than with the
# result. This bounds a request that would otherwise run until the client gives up,
# holding a threadpool slot the entire time.
SCAN_TIMEOUT_SECONDS = 300


class LiveScanTimeout(Exception):
    """
    A live scan exceeded :data:`SCAN_TIMEOUT_SECONDS`.

    Raised rather than returning a partial index: half a listing looks like a complete
    one to the caller, and silently dropping assets is worse than failing.
    """


class _Deadline:
    """
    Wall-clock budget for one scan, checked between S3 calls.

    Cooperative rather than pre-emptive: boto3 calls are not interruptible, so the
    deadline is enforced at each page boundary and before each header read. The
    overshoot is therefore bounded by one S3 request, not unbounded.
    """

    def __init__(self, seconds: float):
        """
        :param seconds: budget from construction
        """

        self.seconds = seconds
        self.started = time.monotonic()

    def elapsed(self) -> float:
        """
        :return: seconds since the scan began
        """

        return time.monotonic() - self.started

    def check(self, what: str):
        """
        Abort the scan if the budget is spent.

        :param what: what was in progress, for the message
        :raises LiveScanTimeout: when the deadline has passed
        """

        if self.elapsed() > self.seconds:
            raise LiveScanTimeout(
                f"Live scan exceeded {self.seconds:.0f}s while {what}; narrow the "
                f"request or raise datascience_results.<domain>.live_scan.timeout_seconds")


class LiveScanNotConfigured(Exception):
    """
    A domain has no S3 scan settings, so a live scan cannot be attempted.

    Raised rather than scanning a default bucket: the domains write to different
    buckets in different regions, and guessing would silently return nothing.
    """


def _s3_client(region_name):
    """
    Build an S3 client for a region, with the service's shared retry/timeout policy.

    :param region_name: AWS region the bucket lives in
    :return: a boto3 S3 client
    """

    return boto3.client('s3', region_name=region_name, config=S3_CLIENT_CONFIG)


def _deadline_for(settings) -> _Deadline:
    """
    Build the scan's budget, honouring a per-domain override.

    :param settings: the domain's live_scan config
    :return: a started :class:`_Deadline`
    """

    return _Deadline(float(settings.get('timeout_seconds') or SCAN_TIMEOUT_SECONDS))


def _scan_settings(domain: str):
    """
    Read a domain's ``live_scan`` block out of configs/config_datascience_results.yaml.

    :param domain: 'multispectral' or 'treetracker'
    :return: the domain's live_scan config
    :raises LiveScanNotConfigured: if the domain has no live_scan block or no bucket
    """

    domain_config = (config.datascience_results.get(domain) or {}).get('live_scan')

    if not domain_config or not domain_config.get('bucket'):
        raise LiveScanNotConfigured(
            f"No live_scan bucket configured for '{domain}'; set "
            f"datascience_results.{domain}.live_scan.bucket in "
            f"configs/config_datascience_results.yaml")

    return domain_config


def _list_keys(client, bucket: str, prefix: str, extensions: tuple, deadline):
    """
    Page through a prefix, yielding objects whose key ends with one of ``extensions``.

    Paginated rather than a single list call: a season's rasters run well past the
    1000-key page limit, and a truncated listing would silently drop results.

    :param client: S3 client for the bucket's region
    :param bucket: bucket to scan
    :param prefix: key prefix to walk
    :param extensions: file suffixes to keep
    :param deadline: budget checked at each page boundary
    :return: iterator of the raw S3 object dicts
    :raises LiveScanTimeout: if the budget is spent mid-listing
    """

    paginator = client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        deadline.check(f"listing s3://{bucket}/{prefix}")

        for obj in page.get('Contents', []):
            if obj['Key'].endswith(extensions):
                yield obj


# ---------------------------------------------------------------------------
# multispectral
# ---------------------------------------------------------------------------

def scan_multispectral_rasters(country=None, site=None, subsite=None,
                               period=None, raster=None):
    """
    Index the multispectral output rasters currently in S3.

    Each key encodes its own project identifiers, so the row set is derived from the
    key layout rather than from a lookup table:
    ``<base>/<country>/<site>/<subsite>/<period>/outputs/rasters/<file>.tif``. Keys
    that do not carry the outputs marker -- input bands, indices, AOIs -- are skipped.

    :param country: optional country filter
    :param site: optional site filter
    :param subsite: optional subsite filter
    :param period: optional period filter
    :param raster: optional raster product filter, matched against the file stem
    :return: list of row dicts, sorted by country/site/subsite/period/raster
    :raises LiveScanNotConfigured: if the domain has no bucket configured
    :raises LiveScanTimeout: if the scan exceeds its configured budget
    """

    settings = _scan_settings('multispectral')

    bucket = settings['bucket']
    base_prefix = settings['base_prefix']
    marker = settings['rasters_marker']
    extensions = tuple(settings.get('extensions') or ['.tif'])
    region = settings.get('region') or os.getenv("AWS_S3_REGION_NAME")

    logger.info(f"live_scan multispectral listing: bucket={bucket} prefix={base_prefix} extensions={extensions}")

    deadline = _deadline_for(settings)
    client = _s3_client(region)
    rows = []

    for obj in _list_keys(client, bucket, base_prefix, extensions, deadline):
        key = obj['Key']

        # Only output rasters carry the marker; everything else is an input layer.
        if marker not in key:
            continue

        segments = key.split(marker, 1)[0].rstrip('/').split('/')

        # The four project identifiers sit immediately before the marker.
        if len(segments) < 4:
            logger.warning(f"live_scan skipping key: reason=unexpected_path key={key}")
            continue

        key_country, key_site, key_subsite, key_period = segments[-4:]
        filename = os.path.basename(key)
        raster_name = os.path.splitext(filename)[0]

        modified = obj.get('LastModified')

        rows.append({
            'country': key_country,
            'site': key_site,
            'subsite': key_subsite,
            'period': key_period,
            'raster': filename,
            'raster_name': raster_name,
            's3_key': key,
            's3_bucket': bucket,
            's3_uri': f"s3://{bucket}/{key}",
            'https_url': f"https://{bucket}.s3.{region}.amazonaws.com/{key}",
            'size_bytes': obj.get('Size'),
            'last_modified': modified.isoformat() if modified else None,
        })

    rows = _apply_filters(rows, {
        'country': country, 'site': site, 'subsite': subsite, 'period': period,
    })

    if raster is not None:
        # Match the product name against the file stem, so 'PRODUCTIVITY' finds
        # PRODUCTIVITY.tif without the caller needing the extension.
        rows = [r for r in rows if raster.lower() in r['raster_name'].lower()]

    rows.sort(key=lambda r: (r['country'], r['site'], r['subsite'],
                             r['period'], r['raster']))

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    for row in rows:
        row['generated_at'] = generated_at

    logger.info(f"live_scan multispectral finished: rasters={len(rows)} duration={deadline.elapsed():.2f}s")

    return rows


# ---------------------------------------------------------------------------
# treetracker
# ---------------------------------------------------------------------------

def device_and_day_of(filename: str):
    """
    Split a track filename into its device prefix and calendar day.

    Files are named ``<DEVICE>_<YYYYMMDD>_<HHMMSS>.geojson``, sometimes with a second
    device token (``BE0F44_0EDD80_20260413_232154``), so the device is every token
    before the 8-digit date -- keeping both rather than merging two devices that
    happen to share a leading token.

    :param filename: the object's file name
    :return: (device, 'YYYYMMDD'), or (None, None) when there is no date token
    """

    stem = filename[:-len('.geojson')] if filename.endswith('.geojson') else filename
    tokens = stem.split('_')

    for index, token in enumerate(tokens):
        if len(token) == 8 and token.isdigit():
            return ('_'.join(tokens[:index]) or None), token

    return None, None


def _extract_properties(blob: bytes):
    """
    Pull the top-level ``properties`` object out of a partial GeoJSON document.

    Matched by brace depth while tracking string state, so a brace inside a filename
    cannot close the object early.

    :param blob: leading bytes of the GeoJSON file
    :return: the parsed properties dict, or None if the block is not complete in blob
    """

    text = blob.decode('utf-8', errors='ignore')

    marker = text.find('"properties"')
    if marker == -1:
        return None

    start = text.find('{', marker + len('"properties"'))
    if start == -1:
        return None

    depth, in_string, escaped = 0, False, False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except ValueError:
                    return None

    return None


def _fetch_properties(client, bucket: str, key: str, probe_bytes: int, size: int):
    """
    Read one track's header properties without downloading the whole file.

    Track files run to tens of megabytes, but the writer emits ``type`` then
    ``properties`` then ``features``, so the block sits near the start: a range
    request for the leading bytes is enough, with a full read only as a fallback.

    :param client: S3 client for the bucket's region
    :param bucket: bucket holding the object
    :param key: object key
    :param probe_bytes: how many leading bytes to request
    :param size: the object's full size, to avoid a pointless second request
    :return: the properties dict, empty when unreadable
    """

    try:
        head = client.get_object(Bucket=bucket, Key=key,
                                 Range=f"bytes=0-{probe_bytes - 1}")['Body'].read()

        properties = _extract_properties(head)
        if properties is not None:
            return properties

        if size <= probe_bytes:
            return {}

        logger.warning(f"live_scan header overflow: probe_bytes={probe_bytes} action=full_read key={key}")
        body = client.get_object(Bucket=bucket, Key=key)['Body'].read()

        return _extract_properties(body) or {}

    except (ClientError, BotoCoreError, ValueError) as e:
        logger.warning(f"live_scan property read failed: key={key} error={e}")
        return {}


def scan_treetracker_assets(country=None, site=None, session_id=None,
                            read_properties=None):
    """
    Index the per-session GeoJSON tracks currently in S3.

    Keys are ``<prefix>/<planting_session_id>/<DEVICE>_<YYYYMMDD>_<HHMMSS>.geojson``,
    so the session, device and capture date come from the key itself. When
    ``read_properties`` is on, each track's header is range-read for its GPS point
    counts and clipping stats -- accurate, but one extra request per asset.

    ``country`` and ``site`` are accepted for interface parity with the other result
    routes; S3 keys carry no such fields, so they are ignored and logged.

    :param country: ignored, see above
    :param site: ignored, see above
    :param session_id: optional planting session filter
    :param read_properties: override the configured header-reading behaviour
    :return: list of row dicts, sorted by session, capture date and device
    :raises LiveScanNotConfigured: if the domain has no bucket configured
    :raises LiveScanTimeout: if the scan exceeds its configured budget
    """

    settings = _scan_settings('treetracker')

    bucket = settings['bucket']
    prefix = settings['prefix'].rstrip('/')
    region = settings.get('region')
    probe_bytes = settings.get('probe_bytes', 8192)
    max_workers = settings.get('max_workers', 8)

    if read_properties is None:
        read_properties = bool(settings.get('read_properties', False))

    if country or site:
        logger.info("live_scan treetracker ignoring filters: "
                    "reason=country_site_not_in_s3_keys use=session_id")

    logger.info(f"Live scan: s3://{bucket}/{prefix}/ "
                f"{'with header properties' if read_properties else ''}")

    deadline = _deadline_for(settings)
    client = _s3_client(region)
    rows = []

    for obj in _list_keys(client, bucket, f"{prefix}/", ('.geojson',), deadline):
        key = obj['Key']
        parts = key[len(prefix):].strip('/').split('/')

        if len(parts) < 2:
            logger.warning(f"live_scan skipping key: reason=no_session_folder key={key}")
            continue

        key_session_id, filename = parts[0], parts[-1]

        if session_id is not None and key_session_id != session_id:
            continue

        device, day = device_and_day_of(filename)
        modified = obj.get('LastModified')

        row = {
            'planting_session_id': key_session_id,
            'device': device,
            'capture_date': f"{day[:4]}-{day[4:6]}-{day[6:]}" if day else None,
            'file_name': filename,
            's3_key': key,
            's3_bucket': bucket,
            's3_uri': f"s3://{bucket}/{key}",
            'size_bytes': obj.get('Size'),
            'last_modified': modified.isoformat() if modified else None,
            'etag': (obj.get('ETag') or '').strip('"') or None,
        }
        row.update({name: None for name in ASSET_PROPERTIES})
        rows.append(row)

    if rows and read_properties:
        def enrich(row):
            # One S3 GET per asset, so the budget is checked per row rather than only
            # per page -- a large index could otherwise blow well past it here.
            deadline.check(f"reading headers under s3://{bucket}/{prefix}/")

            properties = _fetch_properties(client, bucket, row['s3_key'],
                                           probe_bytes, row['size_bytes'])
            for name in ASSET_PROPERTIES:
                if properties.get(name) is not None:
                    row[name] = properties[name]

            sources = properties.get('source_files')
            if isinstance(sources, list):
                row['source_files'] = ';'.join(str(s) for s in sources)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # list() drains the map so a LiveScanTimeout raised in a worker propagates
            # here rather than being swallowed with the discarded results.
            list(pool.map(enrich, rows))

    rows.sort(key=lambda r: (r['planting_session_id'],
                             r['capture_date'] or '',
                             r['device'] or ''))

    logger.info(f"Live scan found {len(rows)} tree-tracker asset(s) "
                f"in {deadline.elapsed():.1f}s")

    return rows


def _apply_filters(rows, filters):
    """
    Keep the rows matching every supplied filter, ignoring the ones left as None.

    :param rows: row dicts to filter
    :param filters: mapping of key to required value
    :return: the matching rows
    """

    active = {key: value for key, value in filters.items() if value is not None}

    if not active:
        return rows

    return [row for row in rows
            if all(row.get(key) == value for key, value in active.items())]
