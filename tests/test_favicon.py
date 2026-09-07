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
