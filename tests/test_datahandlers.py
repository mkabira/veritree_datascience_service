"""
Tests for the connection handlers.

These cover construction and configuration only -- no network. What matters here is
that the engine is built safely and pooled for long-running server use.
"""

import pytest


class TestPostgresEngine:

    def test_special_characters_in_the_password_survive(self):
        """
        A password containing URL-reserved characters must connect, not misparse.

        f-string URL building silently breaks on '@', '/' and ':' -- the host would be
        read out of the middle of the password.
        """
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="svc_user", password="p@ss/w:rd#1")

        url = handler.engine.url
        assert url.host == "db.example.com"
        assert url.password == "p@ss/w:rd#1"
        assert url.database == "analytics"

    def test_password_is_masked_when_rendered(self):
        """The URL is logged at startup; it must not leak the password."""
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="svc_user", password="hunter2")

        rendered = handler.engine.url.render_as_string()
        assert "hunter2" not in rendered
        assert "***" in rendered

    def test_missing_port_falls_back_to_the_postgres_default(self):
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="db.example.com", port=None, database="analytics",
            user="u", password="p")

        assert handler.engine.url.port == 5432

    def test_pool_is_configured_for_a_long_running_server(self):
        """
        RDS drops idle connections; a pooled connection handed out afterwards fails
        the request. pre_ping validates on checkout, recycle caps connection age.
        """
        from src.awskit.datahandlers import RDSPostgresHandler, DB_POOL_SETTINGS

        assert DB_POOL_SETTINGS['pool_pre_ping'] is True
        assert 0 < DB_POOL_SETTINGS['pool_recycle'] <= 3600

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="u", password="p")
        assert handler.engine.pool.size() == DB_POOL_SETTINGS['pool_size']

    def test_constructing_the_engine_opens_no_connection(self):
        """Boot must not depend on the database being reachable."""
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="127.0.0.1", port="1", database="nope", user="u", password="p")
        assert handler.engine.pool.checkedout() == 0


class TestS3Handler:

    def test_reading_from_an_unconnected_bucket_raises_clearly(self):
        from src.awskit.datahandlers import S3StorageHandler

        handler = S3StorageHandler(aws_bucketname=None, aws_accesskey=None,
                                   aws_secretkey=None, aws_regionname=None)

        assert handler.connected() is False
        with pytest.raises(RuntimeError, match="is not connected"):
            handler.read_bytes("some/key.jpg")

    def test_construction_never_raises_on_a_bad_bucket(self):
        """A misconfigured CV bucket must not stop datascience_results from serving."""
        from src.awskit.datahandlers import S3StorageHandler

        S3StorageHandler(aws_bucketname=None, aws_accesskey=None,
                         aws_secretkey=None, aws_regionname=None)

    def test_client_retries_transient_faults(self):
        from src.awskit.datahandlers import S3_CLIENT_CONFIG

        assert S3_CLIENT_CONFIG.retries['max_attempts'] >= 3
        assert S3_CLIENT_CONFIG.connect_timeout is not None
        assert S3_CLIENT_CONFIG.read_timeout is not None
