"""
Test fixtures for the datascience_results routes.

The routes are exercised against a file-backed sqlite database standing in for the
analytics Postgres instance, so the suite runs with no network and no credentials.
An in-memory sqlite URL will NOT work here: pandas' to_sql and the route's read use
different connections from the pool, and each gets its own empty in-memory database.
"""

import os
import sys

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    from src.awskit import datahandlers
    datahandlers.pg_handler.engine = engine

    from src.awskit import datastores
    datastores.pg_handler = datahandlers.pg_handler

    # sqlite has no schema qualifiers; strip them for the duration of the suite
    original_read_sql = datastores._read_sql
    datastores._read_sql = lambda sql, params=None: original_read_sql(
        sql.replace("bioacoustics.", "").replace("multispectral.", ""), params)

    from api.routers import datascience_results

    app = FastAPI()
    app.include_router(datascience_results.router)

    yield TestClient(app)

    datastores._read_sql = original_read_sql


@pytest.fixture
def auth():
    return {"Token": TEST_TOKEN}
