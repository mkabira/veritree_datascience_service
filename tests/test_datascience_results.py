"""Route-level tests for /datascience_results."""

import pytest


BIOACOUSTICS = "/datascience_results/bioacoustics_results/"
MULTISPECTRAL = "/datascience_results/multispectral_results/"
TREETRACKER = "/datascience_results/treetracker_results/"


class TestAuthentication:

    @pytest.mark.parametrize("route", [BIOACOUSTICS, MULTISPECTRAL, TREETRACKER])
    def test_missing_token_is_rejected(self, client, route):
        assert client.get(route).status_code == 401

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


class TestMultispectralResults:

    def test_unfiltered_returns_all_rows(self, client, auth):
        body = client.get(MULTISPECTRAL, headers=auth).json()
        assert body["n_records"] == 2
        assert body["source"] == "multispectral.tbl_raster_results"

    def test_raster_product_filter(self, client, auth):
        body = client.get(f"{MULTISPECTRAL}?raster=PRODUCTIVITY", headers=auth).json()
        assert body["n_records"] == 1
        assert body["results"][0]["s3_uri"] == "s3://x/a.tif"

    def test_multiple_filters_are_combined_with_and(self, client, auth):
        body = client.get(f"{MULTISPECTRAL}?country=kenya&raster=STRUCTURE", headers=auth).json()
        assert body["n_records"] == 1


class TestTreetrackerResults:

    def test_returns_501_until_upstream_publishes_a_table(self, client, auth):
        response = client.get(TREETRACKER, headers=auth)
        assert response.status_code == 501
        assert "not yet published" in response.json()["detail"]


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

        def spy(table, filters, limit, offset):
            captured.update(table=table, limit=limit, offset=offset)
            return original(table, filters, limit, offset)

        monkeypatch.setattr(datastores, "_fetch_page", spy)
        client.get(f"{BIOACOUSTICS}?limit=1&offset=1", headers=auth)

        assert captured["limit"] == 1, "limit must be passed down, not applied in Python"
        assert captured["offset"] == 1

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
