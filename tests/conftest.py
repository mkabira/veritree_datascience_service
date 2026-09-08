"""
Test fixtures for the datascience_results routes.

The routes are exercised against a file-backed sqlite database standing in for the
analytics Postgres instance, so the suite runs with no network and no credentials.
An in-memory sqlite URL will NOT work here: pandas' to_sql and the route's read use
different connections from the pool, and each gets its own empty in-memory database.
"""

import io
import os
import sys
import types

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Heavy-model stubs, installed BEFORE any test module imports the routers.
#
# The computer_vision router loads YOLO weights, a Keras SavedModel and OpenCLIP at
# import time -- seconds of work needing the full CV stack. Stubbing the three model
# modules here lets the REAL router be imported with its real routes, so route-level
# tests exercise genuine request handling.
#
# This must live in conftest (imported by pytest before any test module) rather than
# in per-file fixtures: two files each inserting their own sys.modules stub collide
# depending on collection order, which is exactly the failure this replaces.
# ---------------------------------------------------------------------------

DEFAULT_DETECTION_RESULT = {
    "image": {"width": 1920, "height": 1080},
    "counts": {"number_mangroves": 2, "number_alive": 1,
               "number_dead": 1, "number_unclear": 0},
    "detections": [
        {"detection_id": "d1", "index": 0, "status": "alive",
         "detection_confidence": 0.88, "bbox_xyxy": [854.39, 650.0, 904.03, 807.26],
         "bbox_xywhn": [0.457922, 0.674658, 0.025858, 0.145614],
         "probabilities": {"alive": 0.9999, "dead": 0.0, "unclear": 0.0001}},
        {"detection_id": "d2", "index": 1, "status": "dead",
         "detection_confidence": 0.72, "bbox_xyxy": [100.0, 200.0, 150.0, 300.0],
         "bbox_xywhn": [0.065, 0.231, 0.026, 0.093],
         "probabilities": {"alive": 0.02, "dead": 0.95, "unclear": 0.03}},
    ],
}


def _install_model_stubs():
    detection = types.ModuleType("src.services.computer_vision.survivability_detection")
    detection.load_models = lambda: (None, None)
    detection.detect_survivability = (
        lambda image, session_id=None: DEFAULT_DETECTION_RESULT)
    sys.modules["src.services.computer_vision.survivability_detection"] = detection

    def _stub_load_photo(url, show=False):
        """Stand in for the HTTP fetch; tests that care patch it on the router."""
        from PIL import Image
        return Image.new("RGB", (1920, 1080))

    veritag = types.ModuleType("src.libs.veritag")
    veritag.load_photo = _stub_load_photo
    veritag.load_anthropic_client = lambda api_key=None: None
    veritag.classify_image_anthropic = lambda **kwargs: {}
    sys.modules["src.libs.veritag"] = veritag

    tagging = types.ModuleType("src.services.computer_vision.content_tagging")
    tagging.run_veritag_anthropic = lambda **kwargs: ([], {})
    sys.modules["src.services.computer_vision.content_tagging"] = tagging


_install_model_stubs()

# src/utils/context.py calls load_dotenv() at import, so the developer's real .env
# leaks into the suite. Neutralise anything that could reach live infrastructure --
# without this, constructing a handler opens an actual SSH tunnel to the bastion.
for _name in list(os.environ):
    if _name.startswith(("AWS_RDS_", "AWS_SOURCE_", "AWS_S3_")) or _name in {
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}:
        del os.environ[_name]

os.environ.setdefault("API_ENDPOINT_TOKEN", "test-token")
os.environ.setdefault("AWS_RDS_ENDPOINT", "localhost")
os.environ.setdefault("AWS_RDS_PORT", "5432")
os.environ.setdefault("AWS_RDS_DB", "test")
os.environ.setdefault("AWS_RDS_USER_NAME", "test")
os.environ.setdefault("AWS_RDS_PASSWORD", "test")


TEST_TOKEN = os.environ["API_ENDPOINT_TOKEN"]


BIOACOUSTICS_ROWS = [
    {"index": 0, "code_country": "KEN", "code_site": "kuchi", "recording_year": "2026",
     "prediction": "Ardea alba", "species_richness": 12, "birdbase_esi_mean": float("nan")},
    {"index": 1, "code_country": "TZA", "code_site": "ushongo", "recording_year": "2026",
     "prediction": "Corvus albus", "species_richness": 7, "birdbase_esi_mean": 0.42},
]

RASTER_ROWS = [
    {"project_name": "kenya_kilifi_kuchi_2026", "country": "kenya", "site": "kuchi",
     "subsite": "a", "period": "2026", "raster": "PRODUCTIVITY", "s3_uri": "s3://x/a.tif"},
    {"project_name": "kenya_kilifi_kuchi_2026", "country": "kenya", "site": "kuchi",
     "subsite": "a", "period": "2026", "raster": "STRUCTURE", "s3_uri": "s3://x/b.tif"},
]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "analytics.db"
    engine = create_engine(f"sqlite:///{db_path}")

    pd.DataFrame(BIOACOUSTICS_ROWS).to_sql(
        "tbl_postprocess_site_level_indicies", engine, index=False)
    pd.DataFrame(RASTER_ROWS).to_sql(
        "tbl_raster_results", engine, index=False)

    from src.awskit import datastores

    # Every domain resolves to the same sqlite fixture. get_handler is the seam: the
    # real one builds a per-domain engine from the environment (and may open an SSH
    # tunnel), which a test must never do.
    class FixtureHandler:
        def __init__(self, engine):
            self.engine = engine
            self.tunnel = None

        def ensure_ready(self):
            pass

    original_get_handler = datastores.get_handler
    datastores.get_handler = lambda domain: FixtureHandler(engine)

    # sqlite has no schema qualifiers. The accessors read their table names from module
    # constants, so pointing those at the unqualified fixture tables is enough -- no
    # need to reach into a private function.
    original_tables = (datastores.BIOACOUSTICS_TABLE, datastores.MULTISPECTRAL_TABLE)
    datastores.BIOACOUSTICS_TABLE = "tbl_postprocess_site_level_indicies"
    datastores.MULTISPECTRAL_TABLE = "tbl_raster_results"

    from api.routers import datascience_results

    app = FastAPI()
    app.include_router(datascience_results.router)

    yield TestClient(app)

    datastores.BIOACOUSTICS_TABLE, datastores.MULTISPECTRAL_TABLE = original_tables
    datastores.get_handler = original_get_handler


@pytest.fixture
def auth():
    return {"Token": TEST_TOKEN}


@pytest.fixture
def analyses_client(monkeypatch):
    """
    TestClient for /analyses with llm_chat stubbed.

    The stub is installed on the router module (where the name was imported to),
    not on src.libs.llm, so `from src.libs.llm import llm_chat` still resolves
    to the fake. Tests set `stub.response` to control what the model 'returns'.
    """
    from api.routers import analyses

    class Stub:
        response = "[]"
        calls = []

        def __call__(self, system_prompt, user_prompt, model, temperature):
            Stub.calls.append({"model": model, "temperature": temperature,
                               "user_prompt": user_prompt})
            if isinstance(Stub.response, Exception):
                raise Stub.response
            return Stub.response

    stub = Stub()
    Stub.calls = []
    monkeypatch.setattr(analyses, "llm_chat", stub)

    app = FastAPI()
    app.include_router(analyses.router)

    return TestClient(app), Stub


@pytest.fixture
def rule_payload():
    return {
        "rules": [
            {
                "rule_public_id": "rule_abc123",
                "name": "Photo missing meterstick",
                "status": "failed",
                "failure_reason": {"detail": "no meterstick detected"},
                "flagged": True,
                "comments": [
                    {"source": "field", "actor_id": "u1",
                     "message": "stick was out of frame", "created_at": "2026-03-16"}
                ],
            }
        ]
    }


@pytest.fixture
def cv_client(monkeypatch):
    """
    TestClient for the real computer_vision router, with S3 and the models stubbed.

    Returns (client, FakeS3). Set FakeS3.payload to an Exception to simulate an S3
    failure, or to arbitrary bytes to simulate a corrupt image.
    """
    from PIL import Image
    from api.routers import computer_vision

    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), (34, 80, 40)).save(buffer, "JPEG")

    class FakeS3:
        payload = buffer.getvalue()
        requested = []

        def read_bytes(self, key):
            FakeS3.requested.append(key)
            if isinstance(FakeS3.payload, Exception):
                raise FakeS3.payload
            return FakeS3.payload

    FakeS3.payload = buffer.getvalue()
    FakeS3.requested = []
    monkeypatch.setattr(computer_vision, "s3_source_handler", FakeS3())

    app = FastAPI()
    app.include_router(computer_vision.router)

    return TestClient(app), FakeS3
