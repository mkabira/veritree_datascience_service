"""
veritree datascience service -- FastAPI application entrypoint.

Route groups:
  /computer_vision      inference over field photo evidence
  /datascience_results  read-only results from the analytics database
"""

from datetime import datetime

from fastapi import FastAPI

from api.routers import computer_vision, datascience_results

from src.utils import context


config = context.config
logger = context.logger


app = FastAPI(
    title=config.repo.name,
    description=config.repo.description,
    version=config.repo.version,
)

app.include_router(computer_vision.router)
app.include_router(datascience_results.router)

logger.info(f'veritree datascience service live: {datetime.now()}')


@app.get("/")
async def landing():
    logger.info('client reached: app.get route /')
    return {"response": f"Welcome to the veritree datascience service: {datetime.now()}"}


@app.get("/health")
async def health():
    """Unauthenticated liveness probe for the ECS target group."""
    return {"status": "ok", "service": config.repo.name, "version": config.repo.version}
