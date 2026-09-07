"""
Connection handlers for the AWS services the API depends on.

Exposes an S3 reader for inbound field photos and a shared SQLAlchemy engine for the
analytics Postgres instance. Both are configured for long-running server use: the
database pool recycles and health-checks connections, and the S3 client retries
transient errors rather than surfacing them as request failures.
"""

import os

import boto3
from botocore.config import Config as BotocoreConfig
from sqlalchemy import create_engine
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
            logger.info(f"Created connection to AWS Storage bucket: {aws_bucketname}")

        except Exception as e:
            logger.error(f"Unable to connect to AWS Storage bucket: {aws_bucketname} due to: {e}")

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
            raise RuntimeError(f"S3 bucket '{self.bucket_name}' is not connected")

        return self.s3_bucket.Object(location_source).get()['Body'].read()


class RDSPostgresHandler:
    """
    SQLAlchemy engine for the analytics Postgres instance.

    The engine is lazy: constructing it opens no socket, so the service starts even
    when the database is unreachable and only the datascience_results routes fail.
    """

    def __init__(self, host, port, database, user, password):
        """
        Build a pooled engine for the given instance.

        Credentials are passed to :meth:`sqlalchemy.engine.URL.create` rather than
        interpolated into a URL string: it escapes reserved characters, so a password
        containing '@', '/', ':' or '#' connects correctly instead of misparsing, and
        the resulting URL masks the password when logged or repr'd.

        :param host: RDS endpoint hostname
        :param port: port; coerced to int, defaulting to 5432 when unset
        :param database: database name
        :param user: database user
        :param password: database password
        """

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

        # url renders the password as *** -- safe to log.
        logger.info(f"Configured RDS Postgres engine: {url.render_as_string()}")


# One shared engine backs the datascience_results routes. S3 handlers are constructed
# per-router instead: only computer_vision needs one, and it authenticates with the
# AWS_SOURCE_* credentials rather than the default chain.
pg_handler = RDSPostgresHandler(
    host=os.getenv("AWS_RDS_ENDPOINT"),
    port=os.getenv("AWS_RDS_PORT"),
    database=os.getenv("AWS_RDS_DB"),
    user=os.getenv("AWS_RDS_USER_NAME"),
    password=os.getenv("AWS_RDS_PASSWORD"),
)
