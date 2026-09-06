"""Shared FastAPI dependencies for the veritree datascience service."""

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from src.utils import context


config = context.config
logger = context.logger


async def api_authentication(token: str = Depends(APIKeyHeader(name='Token'))):
    """
    API Authentication.

    Kept on the `Token` header (rather than a bearer scheme) for wire-compatibility
    with the deployed survivability clients.

    :token str: token string for authentication
    """

    if token != os.getenv("API_ENDPOINT_TOKEN"):
        logger.warning("Rejected request with invalid API token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    return None
