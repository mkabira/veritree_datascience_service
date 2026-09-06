import os

import boto3
from sqlalchemy import create_engine

from src.utils import context


config = context.config
logger = context.logger


class S3StorageHandler:
    """
    Connection handler for an S3 bucket.

    Merged from the survivability service (IAM/prod_mode branch, used by the deployed
    ECS task) and the multispectral foresthealth pipeline (extension-filtered listing
    used for raster discovery).
    """

    def __init__(self, aws_bucketname, aws_accesskey, aws_secretkey, aws_regionname, prod_mode=False):

        self.bucket_name = aws_bucketname
        self.s3_bucket = None

        try:
            if prod_mode:
                # Use IAM connection if in PROD mode (based on ECS definitions)
                session = boto3.Session()
            else:
                # Use secret keys and access keys from ENV variables otherwise esp in DEV mode
                session = boto3.Session(aws_access_key_id=aws_accesskey,
                                        aws_secret_access_key=aws_secretkey,
                                        region_name=aws_regionname)

            self.s3_bucket = session.resource('s3').Bucket(aws_bucketname)
            logger.info(f"Created connection to AWS Storage bucket: {aws_bucketname}")

        except Exception as e:
            # Boot is not aborted: a service that only serves datascience_results does not
            # need every CV bucket configured. connected() lets callers check before use.
            logger.error(f'Unable to connect to AWS Storage bucket: {aws_bucketname} due to: {e}')

    def connected(self) -> bool:
        """Whether this handler holds a usable bucket resource."""
        return self.s3_bucket is not None

    def listall(self, ):
        """
        List all objects in the bucket
        """
        for obj in self.s3_bucket.objects.all():
            print(obj.key)
        pass

    def upload(self, location_source, location_destination):
        """
        Upload a file to the S3 Bucket using src and dst paths
        # location_source = 'test/test_downloaded.csv'
        # location_destination = 'test/test_uploaded.csv'
        """

        self.s3_bucket.upload_file(location_source, location_destination)
        # TODO: Add a check if file now exists in destination and return T/F
        return None

    def download(self, location_source, location_destination):
        """
        Download a file to the S3 Bucket using src and dst paths
        # location_source = 'test_uploaded.csv'
        # location_destination = './test_downloaded.csv'
        """

        self.s3_bucket.download_file(location_source, location_destination)
        # TODO: Add a check if file now exists in destination and return T/F
        return None

    def list_objects(self, basepath='survivability/'):
        """
        Returns a list of eligible image objects in the S3 bucket base path
        """

        list_objects = []

        for objects in self.s3_bucket.objects.filter(Prefix=basepath):
            if objects.key.endswith(('.png', '.PNG', 'jpg', 'JPG', 'jpeg', 'JPEG')):
                list_objects.append(objects.key)

        return list_objects

    def list_objects_by_extension(self, basepath='', extensions=('.tif',)):
        """
        Returns a list of object keys under a prefix filtered by file extension
        """

        list_keys = []

        for objects in self.s3_bucket.objects.filter(Prefix=basepath):
            if objects.key.endswith(tuple(extensions)):
                list_keys.append(objects.key)

        return list_keys


class RDSPostgresHandler:

    def __init__(self, host, port, database, user, password):
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
        logger.info(f"Creating connection to RDS Postgres database: {database}")

    def get_engine(self, ):
        """
        Get engine connection for Postgres RDS service using SQLAlchemy
        """
        return self.engine


# ---------------------------------------------------------------------------
# Shared handlers
#
# The datascience_results routers read from a single analytics Postgres instance,
# so one module-level pg_handler is shared. The computer_vision routers talk to
# TWO different buckets (a partner-owned source bucket and a veritree-owned
# destination bucket), so those handlers are constructed in the router itself
# from AWS_SOURCE_* / AWS_DESTINATION_* rather than the single-bucket vars below.
# ---------------------------------------------------------------------------

s3_handler = S3StorageHandler(
    aws_bucketname=os.getenv("AWS_S3_BUCKET_NAME"),
    aws_accesskey=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secretkey=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_regionname=os.getenv("AWS_S3_REGION_NAME"),
    prod_mode=config.prod_mode
)

if s3_handler.connected():
    logger.info(f"Connected to S3 bucket: {s3_handler.bucket_name}")
else:
    logger.warning("S3 handler unavailable; routes that read or write S3 will fail")


pg_handler = RDSPostgresHandler(
    host=os.getenv("AWS_RDS_ENDPOINT"),
    port=os.getenv("AWS_RDS_PORT"),
    database=os.getenv("AWS_RDS_DB"),
    user=os.getenv("AWS_RDS_USER_NAME"),
    password=os.getenv("AWS_RDS_PASSWORD")
)

logger.info("Connected to Postgres")
