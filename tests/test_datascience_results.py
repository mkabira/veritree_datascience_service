"""Route-level tests for /datascience_results."""

import pytest


BIOACOUSTICS = "/datascience_results/bioacoustics_results/"
MULTISPECTRAL = "/datascience_results/multispectral_results/"
TREETRACKER = "/datascience_results/treetracker_results/"


# A MISSING header is rejected by FastAPI's own APIKeyHeader, whose status code is a
# framework detail: 403 up to fastapi 0.115.x, 401 from ~0.14x. Asserting either exact
# code ties the suite to a pinned version, so these check the property that matters --
# the request is refused. A WRONG token is our own code path and is pinned to 401.
UNAUTHENTICATED = {401, 403}


class TestAuthentication:

    @pytest.mark.parametrize("route", [BIOACOUSTICS, MULTISPECTRAL, TREETRACKER])
    def test_missing_token_is_rejected(self, client, route):
        assert client.get(route).status_code in UNAUTHENTICATED

    @pytest.mark.parametrize("route", [BIOACOUSTICS, MULTISPECTRAL, TREETRACKER])
    def test_wrong_token_is_rejected(self, client, route):
        assert client.get(route, headers={"Token": "not-the-token"}).status_code == 401


class TestBioacousticsResults:

    def test_unfiltered_returns_all_rows(self, client, auth):
        body = client.get(BIOACOUSTICS, headers=auth).json()
        assert body["n_records"] == 2
        assert body["source"] == "bioacoustics.tbl_postprocess_site_level_indicies"

    def test_filter_narrows_results(self, client, auth):
        body = client.get(f"{BIOACOUSTICS}?code_country=KEN", headers=auth).json()
        assert body["n_records"] == 1
        assert body["results"][0]["code_site"] == "kuchi"

    def test_filter_with_no_matches_returns_empty(self, client, auth):
        body = client.get(f"{BIOACOUSTICS}?code_site=nowhere", headers=auth).json()
        assert body["n_records"] == 0
        assert body["results"] == []

    def test_internal_index_column_is_dropped(self, client, auth):
        body = client.get(BIOACOUSTICS, headers=auth).json()
        assert "index" not in body["results"][0]

    def test_nan_is_rendered_as_null(self, client, auth):
        """NaN is not valid JSON; it must reach the client as null."""
        body = client.get(f"{BIOACOUSTICS}?code_country=KEN", headers=auth).json()
        assert body["results"][0]["birdbase_esi_mean"] is None

    def test_limit_pages_results_without_changing_the_total(self, client, auth):
        body = client.get(f"{BIOACOUSTICS}?limit=1", headers=auth).json()
        assert body["n_records"] == 2, "n_records is the total matching the filters"
        assert body["n_returned"] == 1
        assert len(body["results"]) == 1

    def test_offset_walks_the_result_set(self, client, auth):
        first = client.get(f"{BIOACOUSTICS}?limit=1&offset=0", headers=auth).json()
        second = client.get(f"{BIOACOUSTICS}?limit=1&offset=1", headers=auth).json()
        assert first["results"][0] != second["results"][0]

    def test_offset_past_the_end_returns_empty(self, client, auth):
        body = client.get(f"{BIOACOUSTICS}?offset=99", headers=auth).json()
        assert body["n_returned"] == 0


SCANNED_RASTERS = [
    {"country": "kenya", "site": "kuchi", "subsite": "a", "period": "2026",
     "raster": "PRODUCTIVITY.tif", "raster_name": "PRODUCTIVITY",
     "s3_key": "p/kenya/kuchi/a/2026/outputs/rasters/PRODUCTIVITY.tif",
     "s3_uri": "s3://bkt/p/kenya/kuchi/a/2026/outputs/rasters/PRODUCTIVITY.tif",
     "https_url": "https://bkt.s3.us-east-2.amazonaws.com/p/PRODUCTIVITY.tif",
     "size_bytes": 1024, "last_modified": "2026-03-16T00:00:00+00:00",
     "generated_at": "2026-09-07T00:00:00+00:00"},
    {"country": "kenya", "site": "kuchi", "subsite": "a", "period": "2026",
     "raster": "STRUCTURE.tif", "raster_name": "STRUCTURE",
     "s3_key": "p/kenya/kuchi/a/2026/outputs/rasters/STRUCTURE.tif",
     "s3_uri": "s3://bkt/p/kenya/kuchi/a/2026/outputs/rasters/STRUCTURE.tif",
     "https_url": "https://bkt.s3.us-east-2.amazonaws.com/p/STRUCTURE.tif",
     "size_bytes": 2048, "last_modified": "2026-03-16T00:00:00+00:00",
     "generated_at": "2026-09-07T00:00:00+00:00"},
]

SCANNED_TRACKS = [
    {"planting_session_id": "12345", "device": "BE0F44", "capture_date": "2026-04-13",
     "file_name": "BE0F44_20260413_232154.geojson",
     "s3_key": "data/data_optimized/12345/BE0F44_20260413_232154.geojson",
     "s3_uri": "s3://trk/data/data_optimized/12345/BE0F44_20260413_232154.geojson",
     "s3_bucket": "trk", "size_bytes": 4096,
     "last_modified": "2026-04-13T23:21:54+00:00", "etag": "abc",
     "gps_point_count": 1200},
]


class TestDatabaseSourceIsNotImplemented:
    """
    The published-table path is deliberately unavailable for these two domains, and
    the 501 must say what to do instead rather than just refusing.
    """

    @pytest.mark.parametrize("route", [MULTISPECTRAL, TREETRACKER])
    def test_without_live_scan_the_route_returns_501(self, client, auth, route):
        response = client.get(route, headers=auth)

        assert response.status_code == 501
        assert "live_scan=true" in response.json()["detail"]

    @pytest.mark.parametrize("route", [MULTISPECTRAL, TREETRACKER])
    def test_live_scan_defaults_to_off(self, client, auth, route):
        """Explicitly opting in keeps the S3 cost visible at the call site."""
        assert client.get(f"{route}?limit=5", headers=auth).status_code == 501


class TestMultispectralLiveScan:

    def _patch(self, monkeypatch, rows=None, error=None):
        from api.routers import datascience_results as module

        captured = {}

        def fake_scan(**kwargs):
            captured.update(kwargs)
            if error:
                raise error
            return rows if rows is not None else SCANNED_RASTERS

        monkeypatch.setattr(module.live_scan_module, "scan_multispectral_rasters", fake_scan)
        return captured

    def test_live_scan_returns_the_indexed_rasters(self, client, auth, monkeypatch):
        self._patch(monkeypatch)
        body = client.get(f"{MULTISPECTRAL}?live_scan=true", headers=auth).json()

        assert body["n_records"] == 2
        assert body["source_type"] == "live_scan"
        assert body["results"][0]["raster_name"] == "PRODUCTIVITY"
        assert body["results"][0]["s3_uri"].startswith("s3://")

    def test_filters_are_passed_to_the_scan(self, client, auth, monkeypatch):
        captured = self._patch(monkeypatch)
        client.get(f"{MULTISPECTRAL}?live_scan=true&country=kenya&site=kuchi"
                   f"&subsite=a&period=2026&raster=PRODUCTIVITY", headers=auth)

        assert captured == {"country": "kenya", "site": "kuchi", "subsite": "a",
                            "period": "2026", "raster": "PRODUCTIVITY"}

    def test_paging_applies_to_the_scanned_rows(self, client, auth, monkeypatch):
        self._patch(monkeypatch)
        body = client.get(f"{MULTISPECTRAL}?live_scan=true&limit=1", headers=auth).json()

        assert body["n_records"] == 2, "the total is every row the scan found"
        assert body["n_returned"] == 1
        assert len(body["results"]) == 1

    def test_an_unconfigured_bucket_is_501(self, client, auth, monkeypatch):
        from src.services.live_scan import LiveScanNotConfigured

        self._patch(monkeypatch, error=LiveScanNotConfigured("no bucket configured"))
        response = client.get(f"{MULTISPECTRAL}?live_scan=true", headers=auth)

        assert response.status_code == 501
        assert "no bucket configured" in response.json()["detail"]

    def test_an_s3_failure_is_502_and_does_not_leak(self, client, auth, monkeypatch):
        self._patch(monkeypatch, error=RuntimeError("AccessDenied on secret-bucket"))
        response = client.get(f"{MULTISPECTRAL}?live_scan=true", headers=auth)

        assert response.status_code == 502
        assert "secret-bucket" not in response.json()["detail"]


class TestTreetrackerLiveScan:

    def _patch(self, monkeypatch, rows=None, error=None):
        from api.routers import datascience_results as module

        captured = {}

        def fake_scan(**kwargs):
            captured.update(kwargs)
            if error:
                raise error
            return rows if rows is not None else SCANNED_TRACKS

        monkeypatch.setattr(module.live_scan_module, "scan_treetracker_assets", fake_scan)
        return captured

    def test_live_scan_returns_the_session_assets(self, client, auth, monkeypatch):
        self._patch(monkeypatch)
        body = client.get(f"{TREETRACKER}?live_scan=true", headers=auth).json()

        assert body["source_type"] == "live_scan"
        assert body["results"][0]["planting_session_id"] == "12345"
        assert body["results"][0]["file_name"].endswith(".geojson")

    def test_session_id_is_passed_to_the_scan(self, client, auth, monkeypatch):
        captured = self._patch(monkeypatch)
        client.get(f"{TREETRACKER}?live_scan=true&session_id=12345", headers=auth)

        assert captured["session_id"] == "12345"

    def test_country_and_site_are_accepted_though_s3_cannot_filter_on_them(
            self, client, auth, monkeypatch):
        """Interface parity with the other routes; the scan logs and ignores them."""
        captured = self._patch(monkeypatch)
        response = client.get(f"{TREETRACKER}?live_scan=true&country=KE&site=kuchi",
                              headers=auth)

        assert response.status_code == 200
        assert captured["country"] == "KE" and captured["site"] == "kuchi"


class TestValidation:

    @pytest.mark.parametrize("query", ["limit=0", "limit=99999", "offset=-1"])
    def test_out_of_range_paging_is_rejected(self, client, auth, query):
        assert client.get(f"{BIOACOUSTICS}?{query}", headers=auth).status_code == 422

    def test_filter_values_are_bound_not_interpolated(self, client, auth):
        """A quote-breaking filter value must match nothing, not widen the result set."""
        body = client.get(f"{BIOACOUSTICS}?code_country=KEN' OR '1'='1", headers=auth).json()
        assert body["n_records"] == 0


class TestPagingHappensInSql:
    """
    Paging must be pushed to the database, not applied to a fully-materialised frame.

    These tables grow every survey season; `select *` followed by a Python slice
    would load the whole table into memory on every request.
    """

    def test_limit_and_offset_reach_the_database(self, client, auth, monkeypatch):
        from src.awskit import datastores

        captured = {}
        original = datastores._fetch_page

        def spy(domain, table, filters, limit, offset):
            captured.update(domain=domain, table=table, limit=limit, offset=offset)
            return original(domain, table, filters, limit, offset)

        monkeypatch.setattr(datastores, "_fetch_page", spy)
        client.get(f"{BIOACOUSTICS}?limit=1&offset=1", headers=auth)

        assert captured["limit"] == 1, "limit must be passed down, not applied in Python"
        assert captured["offset"] == 1
        assert captured["domain"] == "bioacoustics", "each accessor queries its own database"

    def test_only_the_page_is_fetched_not_the_whole_table(self, client, auth):
        """n_records is the full count; the row payload is only the page."""
        body = client.get(f"{BIOACOUSTICS}?limit=1", headers=auth).json()

        assert body["n_records"] == 2, "total comes from a COUNT, not len(rows)"
        assert body["n_returned"] == 1
        assert len(body["results"]) == 1

    def test_the_table_name_is_never_taken_from_request_data(self, client, auth):
        """Table names are interpolated, so they must come from module constants."""
        from src.awskit import datastores
        import inspect

        source = inspect.getsource(datastores)
        assert "BIOACOUSTICS_TABLE" in source
        assert "MULTISPECTRAL_TABLE" in source
        # the only f-string interpolations into SQL are the table and the where clause
        for line in source.splitlines():
            if 'text(f"' in line:
                assert '{table}' in line or '{where}' in line, f"unbound SQL: {line.strip()}"


class TestUnpublishedTables:
    """
    An upstream table that has not been published yet is not a server error.

    bioacoustics results live in a separate database today; treetracker publishes no
    results table at all. Both should read as "not available yet", distinguishable
    from a genuine query failure.
    """

    def test_a_missing_table_is_501_not_500(self, client, auth, monkeypatch):
        from src.awskit import datastores

        def missing(*args, **kwargs):
            raise datastores.ResultsTableUnavailable(
                "'bioacoustics.tbl_postprocess_site_level_indicies' has not been "
                "published to this database yet")

        monkeypatch.setattr(datastores, "get_postprocess_site_level_indicies", missing)

        response = client.get(BIOACOUSTICS, headers=auth)
        assert response.status_code == 501
        assert "has not been published" in response.json()["detail"]

    def test_unavailable_is_a_notimplementederror(self):
        """The routes catch NotImplementedError; the subclass must stay one."""
        from src.awskit.datastores import ResultsTableUnavailable

        assert issubclass(ResultsTableUnavailable, NotImplementedError)

    def test_a_real_query_failure_does_not_leak_sql(self, client, auth, monkeypatch):
        """The DB error names tables and columns; log it, don't return it."""
        from src.awskit import datastores

        def boom(*args, **kwargs):
            raise RuntimeError('syntax error at "select * from secret_table"')

        monkeypatch.setattr(datastores, "get_postprocess_site_level_indicies", boom)

        response = client.get(BIOACOUSTICS, headers=auth)
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "secret_table" not in detail
        assert "see the service logs" in detail
