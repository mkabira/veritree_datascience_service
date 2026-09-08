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

    def test_an_unconfigured_bucket_names_the_env_vars_to_set(self):
        """The common misconfiguration; the error must say how to fix it."""
        from src.awskit.datahandlers import S3StorageHandler

        handler = S3StorageHandler(aws_bucketname=None, aws_accesskey=None,
                                   aws_secretkey=None, aws_regionname=None)

        assert handler.connected() is False
        with pytest.raises(RuntimeError, match="AWS_SOURCE_S3_BUCKET_NAME"):
            handler.read_bytes("some/key.jpg")

    def test_a_named_but_unreachable_bucket_reports_the_bucket(self):
        from src.awskit.datahandlers import S3StorageHandler

        handler = S3StorageHandler(aws_bucketname="some-bucket", aws_accesskey=None,
                                   aws_secretkey=None, aws_regionname=None)
        handler.s3_bucket = None   # simulate a failed connection

        with pytest.raises(RuntimeError, match="some-bucket"):
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


class TestSshTunnel:
    """
    The tunnel is opt-in: AWS_RDS_SSH_HOST is the switch. These cover the wiring and
    the failure modes; the end-to-end path is exercised against a real sshd manually.
    """

    def test_no_ssh_host_means_a_direct_connection(self):
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="u", password="p")

        assert handler.tunnel is None
        assert handler.engine.url.host == "db.example.com"

    def test_ensure_ready_is_a_noop_without_a_tunnel(self):
        from src.awskit.datahandlers import RDSPostgresHandler

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="u", password="p")

        handler.ensure_ready()   # must not raise
        handler.close()          # must not raise

    def test_a_missing_private_key_fails_with_the_path(self, tmp_path):
        from src.awskit.datahandlers import RDSPostgresHandler

        with pytest.raises(RuntimeError, match="SSH private key not found"):
            RDSPostgresHandler(
                host="db.example.com", port="5432", database="analytics",
                user="u", password="p",
                ssh_host="bastion.example.com", ssh_user="ec2-user",
                ssh_pkey=str(tmp_path / "absent_id_rsa"))

    def test_an_unreachable_bastion_names_the_hop(self, tmp_path):
        """The error must say which hop failed, not just 'connection refused'."""
        from src.awskit.datahandlers import RDSPostgresHandler

        key = tmp_path / "id_rsa"
        key.write_text("not a real key")

        with pytest.raises(RuntimeError, match="Unable to open SSH tunnel"):
            RDSPostgresHandler(
                host="db.example.com", port="5432", database="analytics",
                user="u", password="p",
                ssh_host="127.0.0.1", ssh_port="1", ssh_user="nobody",
                ssh_pkey=str(key))

    def test_a_restarted_tunnel_rebinds_a_new_port(self, monkeypatch):
        """
        A restart binds a DIFFERENT local port, so the engine must resolve it per
        connection rather than from the URL, and the stale pool must be dropped.
        """
        from src.awskit.datahandlers import RDSPostgresHandler

        class FakeTunnel:
            def __init__(self):
                self.local_bind_port = 40000
                self.is_active = True
                self.restarted = False

            def restart(self):
                self.local_bind_port = 40001   # a new port, as sshtunnel does
                self.is_active = True
                self.restarted = True

            def stop(self):
                self.is_active = False

        handler = RDSPostgresHandler(
            host="db.example.com", port="5432", database="analytics",
            user="u", password="p")
        handler.tunnel = FakeTunnel()

        disposed = []
        monkeypatch.setattr(handler.engine, "dispose", lambda: disposed.append(True))

        handler.ensure_ready()
        assert handler.tunnel.restarted is False, "a healthy tunnel must not restart"
        assert disposed == []

        handler.tunnel.is_active = False
        handler.ensure_ready()

        assert handler.tunnel.restarted is True
        assert handler.tunnel.local_bind_port == 40001
        assert disposed == [True], "the pool must be dropped; its sockets are stale"


class TestPerDomainEngines:
    """
    Each result domain has its own database on its own server. Settings resolve from
    the domain's prefix first, then the shared AWS_RDS_* names, so a single-database
    deployment needs no per-domain variables at all.
    """

    def test_connection_settings_come_only_from_the_domain_prefix(self, monkeypatch):
        from src.awskit.datahandlers import domain_settings

        monkeypatch.setenv("AWS_RDS_BIOACOUSTICS_ENDPOINT", "bio.example.com")
        monkeypatch.setenv("AWS_RDS_BIOACOUSTICS_DB", "bioacoustics")

        settings = domain_settings("bioacoustics")
        assert settings["host"] == "bio.example.com"
        assert settings["database"] == "bioacoustics"

    @pytest.mark.parametrize("suffix,key", [
        ("ENDPOINT", "host"), ("PORT", "port"), ("DB", "database"),
        ("USER_NAME", "user"), ("PASSWORD", "password"),
    ])
    def test_a_shared_connection_name_is_NOT_a_fallback(self, monkeypatch, suffix, key):
        """
        Falling back would point a domain at another domain's server and query an
        instance that does not hold its schema. Only the SSH settings are shared.
        """
        from src.awskit.datahandlers import domain_settings

        monkeypatch.setenv(f"AWS_RDS_{suffix}", "shared-value")
        monkeypatch.delenv(f"AWS_RDS_MULTISPECTRAL_{suffix}", raising=False)

        assert domain_settings("multispectral")[key] is None

    def test_domains_resolve_independently(self, monkeypatch):
        """The bug this replaces: one endpoint could only serve one domain."""
        from src.awskit.datahandlers import domain_settings

        monkeypatch.setenv("AWS_RDS_BIOACOUSTICS_ENDPOINT", "bio.example.com")
        monkeypatch.setenv("AWS_RDS_MULTISPECTRAL_ENDPOINT", "multi.example.com")

        assert domain_settings("bioacoustics")["host"] == "bio.example.com"
        assert domain_settings("multispectral")["host"] == "multi.example.com"

    def test_ssh_settings_DO_fall_back_to_the_shared_names(self, monkeypatch):
        """One bastion usually fronts every database, so these are shared by default."""
        from src.awskit.datahandlers import domain_settings

        monkeypatch.setenv("AWS_RDS_SSH_HOST", "shared-bastion")
        monkeypatch.setenv("AWS_RDS_SSH_USER", "ec2-user")
        monkeypatch.delenv("AWS_RDS_MULTISPECTRAL_SSH_HOST", raising=False)

        settings = domain_settings("multispectral")
        assert settings["ssh_host"] == "shared-bastion"
        assert settings["ssh_user"] == "ec2-user"

    def test_ssh_settings_can_still_be_overridden_per_domain(self, monkeypatch):
        from src.awskit.datahandlers import domain_settings

        monkeypatch.setenv("AWS_RDS_SSH_HOST", "shared-bastion")
        monkeypatch.setenv("AWS_RDS_BIOACOUSTICS_SSH_HOST", "bio-bastion")

        assert domain_settings("bioacoustics")["ssh_host"] == "bio-bastion"
        assert domain_settings("multispectral")["ssh_host"] == "shared-bastion"

    def test_an_unconfigured_domain_says_which_variable_to_set(self, monkeypatch):
        from src.awskit.datahandlers import DatabaseNotConfigured, get_handler
        import src.awskit.datahandlers as dh

        monkeypatch.setattr(dh, "_handlers", {})
        monkeypatch.delenv("AWS_RDS_TREETRACKER_ENDPOINT", raising=False)

        with pytest.raises(DatabaseNotConfigured, match="AWS_RDS_TREETRACKER_ENDPOINT"):
            get_handler("treetracker")

    def test_handlers_are_built_once_and_cached(self, monkeypatch):
        """An unused domain must never open a connection or an SSH tunnel."""
        from src.awskit.datahandlers import get_handler
        import src.awskit.datahandlers as dh

        monkeypatch.setattr(dh, "_handlers", {})
        for name, value in [("AWS_RDS_BIOACOUSTICS_ENDPOINT", "bio.example.com"),
                            ("AWS_RDS_BIOACOUSTICS_DB", "analytics"),
                            ("AWS_RDS_BIOACOUSTICS_USER_NAME", "u"),
                            ("AWS_RDS_BIOACOUSTICS_PASSWORD", "p")]:
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("AWS_RDS_BIOACOUSTICS_SSH_HOST", raising=False)
        monkeypatch.delenv("AWS_RDS_SSH_HOST", raising=False)

        first = get_handler("bioacoustics")
        assert get_handler("bioacoustics") is first
        assert "multispectral" not in dh._handlers, "unused domains must stay unbuilt"

    def test_a_failure_is_not_cached(self, monkeypatch):
        """Fixing the environment and retrying must work without a restart."""
        from src.awskit.datahandlers import DatabaseNotConfigured, get_handler
        import src.awskit.datahandlers as dh

        monkeypatch.setattr(dh, "_handlers", {})
        for name in ["AWS_RDS_MULTISPECTRAL_ENDPOINT", "AWS_RDS_SSH_HOST",
                     "AWS_RDS_MULTISPECTRAL_SSH_HOST"]:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(DatabaseNotConfigured):
            get_handler("multispectral")

        for name, value in [("AWS_RDS_MULTISPECTRAL_ENDPOINT", "multi.example.com"),
                            ("AWS_RDS_MULTISPECTRAL_DB", "analytics"),
                            ("AWS_RDS_MULTISPECTRAL_USER_NAME", "u")]:
            monkeypatch.setenv(name, value)
        assert get_handler("multispectral").engine.url.host == "multi.example.com"

    @pytest.mark.parametrize("missing,expected", [
        ({"AWS_RDS_ENDPOINT"}, "ENDPOINT"),
        ({"AWS_RDS_DB"}, "_DB"),
        ({"AWS_RDS_USER_NAME"}, "USER_NAME"),
    ])
    def test_every_missing_setting_is_named(self, monkeypatch, missing, expected):
        """
        Without this the failure surfaces as psycopg2's 'fe_sendauth: no password
        supplied', which says nothing about which variable to set.
        """
        from src.awskit.datahandlers import DatabaseNotConfigured, get_handler
        import src.awskit.datahandlers as dh

        monkeypatch.setattr(dh, "_handlers", {})
        for name, value in [("AWS_RDS_BIOACOUSTICS_ENDPOINT", "db.example.com"),
                            ("AWS_RDS_BIOACOUSTICS_DB", "analytics"),
                            ("AWS_RDS_BIOACOUSTICS_USER_NAME", "u"),
                            ("AWS_RDS_BIOACOUSTICS_PASSWORD", "p")]:
            monkeypatch.setenv(name, value)
        for name in ["AWS_RDS_SSH_HOST", "AWS_RDS_BIOACOUSTICS_SSH_HOST"]:
            monkeypatch.delenv(name, raising=False)
        for name in missing:
            monkeypatch.delenv(name.replace("AWS_RDS_", "AWS_RDS_BIOACOUSTICS_"), raising=False)

        with pytest.raises(DatabaseNotConfigured, match=expected):
            get_handler("bioacoustics")
