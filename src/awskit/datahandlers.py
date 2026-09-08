"""
Connection handlers for the AWS services the API depends on.

Exposes an S3 reader for inbound field photos and a shared SQLAlchemy engine for the
analytics Postgres instance. Both are configured for long-running server use: the
database pool recycles and health-checks connections, and the S3 client retries
transient errors rather than surfacing them as request failures.
"""

import atexit
import os
import threading

import boto3
from botocore.config import Config as BotocoreConfig
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL

from src.utils import context


config = context.config
logger = context.logger


# Retry transient S3 faults (throttling, 5xx) before failing a request. Standard mode
# covers the connection-level errors that adaptive mode's client-side rate limiting
# would otherwise mask behind added latency.
S3_CLIENT_CONFIG = BotocoreConfig(
    retries={'max_attempts': 3, 'mode': 'standard'},
    connect_timeout=5,
    read_timeout=30,
)

# RDS terminates idle connections, and a pooled connection handed out after that has
# happened fails the request. pre_ping validates on checkout, recycle caps connection
# age below any server-side idle timeout.
DB_POOL_SETTINGS = {
    'pool_size': 5,
    'max_overflow': 5,
    'pool_pre_ping': True,
    'pool_recycle': 1800,
    'pool_timeout': 30,
}

DB_CONNECT_TIMEOUT_SECONDS = 10

# Bound any single query server-side so one pathological request cannot hold a
# connection open indefinitely.
DB_STATEMENT_TIMEOUT_MS = 30_000

# Bastion hosts drop idle sessions; keepalives hold the forward open between requests.
SSH_KEEPALIVE_SECONDS = 30.0


class S3StorageHandler:
    """
    Read-only connection handler for a single S3 bucket.

    The service reads source images and writes nothing back, so only :meth:`read_bytes`
    is exposed. Construction never raises: a bucket that cannot be reached leaves the
    handler unconnected and reads fail per-request, so a misconfigured computer_vision
    bucket does not stop the datascience_results routes from serving.
    """

    def __init__(self, aws_bucketname, aws_accesskey, aws_secretkey, aws_regionname,
                 prod_mode=False):
        """
        Open a session against the named bucket.

        :param aws_bucketname: bucket to read from
        :param aws_accesskey: access key id, ignored when prod_mode is True
        :param aws_secretkey: secret access key, ignored when prod_mode is True
        :param aws_regionname: region, ignored when prod_mode is True
        :param prod_mode: when True authenticate via the ambient credential chain
            (the ECS task role) instead of explicit keys
        """

        self.bucket_name = aws_bucketname
        self.s3_bucket = None

        try:
            if prod_mode:
                # Resolve credentials from the ambient chain: the ECS task role in
                # deployment, environment variables locally.
                session = boto3.Session()
            else:
                session = boto3.Session(aws_access_key_id=aws_accesskey,
                                        aws_secret_access_key=aws_secretkey,
                                        region_name=aws_regionname)

            self.s3_bucket = session.resource('s3', config=S3_CLIENT_CONFIG).Bucket(aws_bucketname)
            logger.info(f"s3 connected: bucket={aws_bucketname} prod_mode={prod_mode}")

        except Exception as e:
            logger.error(f"s3 connect failed: bucket={aws_bucketname} error={e}")

    def connected(self) -> bool:
        """
        Whether the handler holds a usable bucket resource.

        :return: True when reads can be attempted
        """

        return self.s3_bucket is not None

    def read_bytes(self, location_source: str) -> bytes:
        """
        Read an object straight into memory, without a temporary file.

        :param location_source: object key, e.g. 'survivability/org_1/img.jpg'
        :return: the object's raw bytes
        :raises RuntimeError: if the handler never connected to its bucket
        """

        if not self.connected():
            if not self.bucket_name:
                raise RuntimeError(
                    "No S3 bucket is configured: set AWS_SOURCE_S3_BUCKET_NAME (or "
                    "AWS_S3_BUCKET_NAME) in the environment, or address the object "
                    "with a full s3://bucket/key URI")

            raise RuntimeError(
                f"S3 bucket '{self.bucket_name}' is not connected; check the bucket "
                f"name and that credentials are readable")

        return self.s3_bucket.Object(location_source).get()['Body'].read()


class RDSPostgresHandler:
    """
    SQLAlchemy engine for the analytics Postgres instance.

    The engine is lazy: constructing it opens no socket, so the service starts even
    when the database is unreachable and only the datascience_results routes fail.

    When ``ssh_host`` is given, connections are forwarded through an SSH tunnel to
    that bastion and the engine is pointed at the tunnel's local end instead of the
    database host directly. Leave it unset to connect straight to the database.
    """

    def __init__(self, host, port, database, user, password,
                 ssh_host=None, ssh_port=None, ssh_user=None,
                 ssh_pkey=None, ssh_pkey_password=None):
        """
        Build a pooled engine for the given instance, optionally via an SSH tunnel.

        Credentials are passed to :meth:`sqlalchemy.engine.URL.create` rather than
        interpolated into a URL string: it escapes reserved characters, so a password
        containing '@', '/', ':' or '#' connects correctly instead of misparsing, and
        the resulting URL masks the password when logged or repr'd.

        :param host: database endpoint hostname, as resolved from the bastion when
            tunnelling
        :param port: database port; coerced to int, defaulting to 5432 when unset
        :param database: database name
        :param user: database user
        :param password: database password
        :param ssh_host: bastion hostname; when set, all traffic is tunnelled
        :param ssh_port: bastion SSH port, defaulting to 22
        :param ssh_user: SSH username on the bastion
        :param ssh_pkey: path to the private key, e.g. ~/.ssh/id_rsa
        :param ssh_pkey_password: passphrase, if the key is encrypted
        :raises RuntimeError: if tunnelling is requested but cannot be established
        """

        self.tunnel = None
        # Guards tunnel restarts. Handlers run in a threadpool, so several requests can
        # find the tunnel dead at once; without this each would call restart() and
        # dispose() concurrently on the same objects.
        self._tunnel_lock = threading.Lock()
        self._db_host = host
        self._db_port = int(port) if port else 5432

        if ssh_host:
            self.tunnel = self._open_tunnel(
                ssh_host=ssh_host,
                ssh_port=int(ssh_port) if ssh_port else 22,
                ssh_user=ssh_user,
                ssh_pkey=ssh_pkey,
                ssh_pkey_password=ssh_pkey_password)

            # Point the engine at the tunnel's local end. The database host and port
            # are resolved on the bastion's side of the forward.
            host, port = '127.0.0.1', self.tunnel.local_bind_port
            atexit.register(self.close)

        url = URL.create(
            drivername="postgresql+psycopg2",
            username=user,
            password=password,
            host=host,
            port=int(port) if port else 5432,
            database=database,
        )

        self.engine = create_engine(
            url,
            connect_args={
                'connect_timeout': DB_CONNECT_TIMEOUT_SECONDS,
                'options': f'-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}',
                'application_name': config.repo.name,
            },
            **DB_POOL_SETTINGS,
        )

        if self.tunnel is not None:
            # The URL's port is only a placeholder. A restarted tunnel binds a DIFFERENT
            # local port, so the live one is injected per connection -- baking it into
            # the URL leaves every post-restart connection refused on the stale port.
            @event.listens_for(self.engine, "do_connect")
            def _use_current_tunnel_port(dialect, conn_rec, cargs, cparams):
                cparams["host"] = "127.0.0.1"
                cparams["port"] = self.tunnel.local_bind_port

        # url renders the password as *** -- safe to log.
        logger.info(f"postgres engine configured: url={url.render_as_string()}")

    def _open_tunnel(self, ssh_host, ssh_port, ssh_user, ssh_pkey, ssh_pkey_password):
        """
        Forward a local port to the database through the bastion.

        :param ssh_host: bastion hostname
        :param ssh_port: bastion SSH port
        :param ssh_user: SSH username
        :param ssh_pkey: path to the private key
        :param ssh_pkey_password: passphrase, if the key is encrypted
        :return: the started SSHTunnelForwarder
        :raises RuntimeError: if sshtunnel is not installed, the key is missing, or
            the forward cannot be established
        """

        try:
            from sshtunnel import SSHTunnelForwarder
        except ImportError as e:
            raise RuntimeError(
                "AWS_RDS_SSH_HOST is set but the 'sshtunnel' package is not installed; "
                "add it to requirements.txt or unset the AWS_RDS_SSH_* variables") from e

        key_path = os.path.expanduser(ssh_pkey) if ssh_pkey else None

        if key_path and not os.path.exists(key_path):
            raise RuntimeError(f"SSH private key not found at {key_path} "
                               f"(from AWS_RDS_SSH_PKEY)")

        try:
            tunnel = SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username=ssh_user,
                ssh_pkey=key_path,
                ssh_private_key_password=ssh_pkey_password,
                remote_bind_address=(self._db_host, self._db_port),
                # 0 lets the OS pick a free local port, so concurrent workers on one
                # host do not collide on a hardcoded one.
                local_bind_address=('127.0.0.1', 0),
                set_keepalive=SSH_KEEPALIVE_SECONDS,
            )
            tunnel.start()

        except Exception as e:
            raise RuntimeError(
                f"Unable to open SSH tunnel to {ssh_user}@{ssh_host}:{ssh_port} "
                f"forwarding to {self._db_host}:{self._db_port}: {e}") from e

        logger.info(f"ssh tunnel open: local_port={tunnel.local_bind_port} "
                    f"bastion={ssh_user}@{ssh_host}:{ssh_port} "
                    f"target={self._db_host}:{self._db_port}")

        return tunnel

    def ensure_ready(self):
        """
        Restart the tunnel if it has dropped.

        Called before checking a connection out of the pool. A bastion that closes the
        session leaves the pool handing out sockets to a dead forward, which
        pool_pre_ping cannot repair on its own because the local port is gone.
        Cheap when the tunnel is healthy -- an unsynchronised flag read -- and a no-op
        when not tunnelling at all. The restart itself is serialised and re-checked
        under a lock, because handlers run concurrently in a threadpool. After a
        restart the connection pool is disposed, since its sockets point at the
        previous local port.

        :return: None
        """

        if self.tunnel is None or self.tunnel.is_active:
            return

        with self._tunnel_lock:
            # Re-check inside the lock: while this thread waited, another may already
            # have restarted the tunnel, and restarting twice would strand the
            # connections the first restart just established.
            if self.tunnel.is_active:
                return

            logger.warning("ssh tunnel dropped: action=restarting")
            self.tunnel.restart()

            # Pooled connections still reference the old local port, which no longer
            # exists. Drop them so the next checkout dials the new one.
            self.engine.dispose()

            logger.info(f"ssh tunnel restarted: local_port={self.tunnel.local_bind_port}")

    def close(self):
        """
        Close the tunnel, if one is open. Registered with atexit when tunnelling.

        :return: None
        """

        if self.tunnel is not None and self.tunnel.is_active:
            self.tunnel.stop()
            logger.info("ssh tunnel closed")


# One shared engine backs the datascience_results routes. S3 handlers are constructed
# per-router instead: only computer_vision needs one, and it authenticates with the
# AWS_SOURCE_* credentials rather than the default chain.
class DatabaseNotConfigured(Exception):
    """
    No connection settings exist for a result domain.

    Raised instead of connecting to the wrong database: the domains live on separate
    servers, so falling back silently would query an instance that simply does not
    hold that schema.
    """


# Each result domain has its own database, on its own server, with its own
# credentials -- bioacoustics on the read replica, multispectral on the impact-team
# instance.
DOMAIN_ENV_PREFIXES = {
    'bioacoustics': 'AWS_RDS_BIOACOUSTICS',
    'multispectral': 'AWS_RDS_MULTISPECTRAL',
    'treetracker': 'AWS_RDS_TREETRACKER',
}

# Where the database lives. Read ONLY from the domain's own prefix: there is no
# shared default, because falling back would point a domain at another domain's
# server and query an instance that does not hold its schema.
CONNECTION_SETTINGS = {
    'host': 'ENDPOINT',
    'port': 'PORT',
    'database': 'DB',
    'user': 'USER_NAME',
    'password': 'PASSWORD',
}

# How to reach it. These DO fall back to the shared AWS_RDS_SSH_* names: the same
# bastion typically fronts every database, so repeating the host and key per domain
# is noise. Override per domain only when a database sits behind a different jump host.
SHARED_SSH_SETTINGS = {
    'ssh_host': 'SSH_HOST',
    'ssh_port': 'SSH_PORT',
    'ssh_user': 'SSH_USER',
    'ssh_pkey': 'SSH_PKEY',
    'ssh_pkey_password': 'SSH_PKEY_PASSWORD',
}


def domain_settings(domain: str) -> dict:
    """
    Resolve one domain's connection settings from the environment.

    Connection settings come from ``AWS_RDS_<DOMAIN>_*`` alone. SSH settings come from
    ``AWS_RDS_<DOMAIN>_SSH_*`` if present, otherwise the shared ``AWS_RDS_SSH_*``.

    :param domain: one of the keys of :data:`DOMAIN_ENV_PREFIXES`
    :return: kwargs for :class:`RDSPostgresHandler`
    :raises KeyError: if the domain is unknown
    """

    prefix = DOMAIN_ENV_PREFIXES[domain]

    settings = {key: os.getenv(f"{prefix}_{suffix}")
                for key, suffix in CONNECTION_SETTINGS.items()}

    settings.update({key: (os.getenv(f"{prefix}_{suffix}") or os.getenv(f"AWS_RDS_{suffix}"))
                     for key, suffix in SHARED_SSH_SETTINGS.items()})

    return settings


_handlers = {}


def get_handler(domain: str) -> RDSPostgresHandler:
    """
    Get the connection handler for a result domain, building it on first use.

    Construction is lazy and cached: an unused domain never opens a connection or an
    SSH tunnel, and a configured one is built once per process. Failures are not
    cached, so fixing the environment and retrying works without a restart.

    :param domain: one of the keys of :data:`DOMAIN_ENV_PREFIXES`
    :return: the domain's handler
    :raises DatabaseNotConfigured: if the domain has no endpoint configured
    :raises KeyError: if the domain is unknown
    """

    if domain in _handlers:
        return _handlers[domain]

    settings = domain_settings(domain)

    # host/database/user are all required to connect. Checking them together turns
    # what would otherwise surface as psycopg2's "fe_sendauth: no password supplied"
    # into a message naming the variables to set.
    prefix = DOMAIN_ENV_PREFIXES[domain]
    required = {'host': 'ENDPOINT', 'database': 'DB', 'user': 'USER_NAME'}
    missing = [suffix for key, suffix in required.items() if not settings[key]]

    if missing:
        names = ", ".join(f"{prefix}_{suffix}" for suffix in missing)
        raise DatabaseNotConfigured(
            f"No database configured for '{domain}': set {names}")

    logger.info(f"postgres domain connecting: domain={domain} host={settings['host']}")
    _handlers[domain] = RDSPostgresHandler(**settings)

    return _handlers[domain]
