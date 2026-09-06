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
