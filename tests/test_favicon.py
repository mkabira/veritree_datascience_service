"""
Tests for favicon delivery.

Everything under test here lives in api/main.py; conftest stubs the model modules
the CV router loads at import time.
"""

import os

import pytest
from fastapi.testclient import TestClient

from src.utils import context


@pytest.fixture(scope="module")
def app_client():
    """The real application; conftest stubs the models the CV router loads."""
    from api.main import app
    return TestClient(app)


class TestFaviconAsset:

    def test_the_icon_file_is_present(self):
        path = os.path.join(context.root_dir, 'api', 'static', 'favicon.ico')
        assert os.path.exists(path), "api/static/favicon.ico is missing from the repo"

    def test_the_icon_is_a_valid_multi_size_ico(self):
        from PIL import Image
        path = os.path.join(context.root_dir, 'api', 'static', 'favicon.ico')

        with open(path, 'rb') as f:
            assert f.read(4) == b'\x00\x00\x01\x00', "not an ICO file"

        sizes = Image.open(path).info.get('sizes', set())
        assert (16, 16) in sizes and (32, 32) in sizes, f"missing standard sizes: {sizes}"


class TestFaviconRoute:

    def test_favicon_is_served(self, app_client):
        response = app_client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/x-icon"
        assert response.content[:4] == b'\x00\x00\x01\x00'

    def test_favicon_needs_no_auth(self, app_client):
        """Browsers request it without headers; requiring a token would 401 every tab."""
        assert app_client.get("/favicon.ico").status_code == 200

    def test_favicon_is_hidden_from_the_schema(self, app_client):
        schema = app_client.get("/openapi.json").json()
        assert "/favicon.ico" not in schema["paths"]


class TestDocsUseOurFavicon:

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_docs_pages_render(self, app_client, path):
        assert app_client.get(path).status_code == 200

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_docs_link_our_icon_not_fastapis(self, app_client, path):
        """FastAPI's default docs hardcode its own CDN favicon; we override both."""
        body = app_client.get(path).text
        assert "/favicon.ico" in body
        assert "fastapi.tiangolo.com/img/favicon" not in body

    def test_openapi_schema_still_served(self, app_client):
        """Overriding the docs routes must not disturb the schema they read."""
        assert app_client.get("/openapi.json").status_code == 200


class TestHealth:
    """The ECS target group reads this; its shape is a contract."""

    def test_reports_service_identity_and_version(self, app_client):
        body = app_client.get("/health").json()

        assert body["status"] == "ok"
        assert body["service"] == context.config.repo.name
        assert body["version"] == context.config.repo.version

    def test_includes_a_parseable_timestamp(self, app_client):
        import datetime

        body = app_client.get("/health").json()
        parsed = datetime.datetime.fromisoformat(body["timestamp"])

        now = datetime.datetime.now(datetime.timezone.utc)
        assert abs((now - parsed).total_seconds()) < 120

    def test_the_timestamp_carries_its_offset(self, app_client):
        """
        A naive timestamp is ambiguous: a probe is read from wherever the task runs,
        so the reader cannot infer the zone.
        """
        import datetime

        body = app_client.get("/health").json()
        parsed = datetime.datetime.fromisoformat(body["timestamp"])

        assert parsed.tzinfo is not None, "timestamp must be timezone-aware"
        assert parsed.utcoffset() == datetime.timedelta(0), "should be UTC"

    def test_the_timestamp_includes_date_and_time(self, app_client):
        body = app_client.get("/health").json()

        date_part, _, time_part = body["timestamp"].partition("T")
        assert len(date_part.split("-")) == 3
        assert time_part, "timestamp must carry a time, not just a date"

    def test_needs_no_auth(self, app_client):
        """The probe sends no headers; requiring a token would fail every check."""
        assert app_client.get("/health").status_code == 200
