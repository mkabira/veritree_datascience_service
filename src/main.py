"""
Command-line entry point for the veritree datascience service.

Dispatches named tasks and pipelines -- the convention shared across the veritree
datascience repositories, and the contract the ECS task definitions invoke:

    python3 src/main.py pipeline pipeline_api
    python3 src/main.py task task_launch_api
    python3 src/main.py list

A *task* is one unit of work; a *pipeline* is an ordered list of tasks run in sequence.
Today the only task launches the API, and scheduled work belongs in the same registry.

Import note: this module is run as a script, so ``sys.path[0]`` is ``src/`` and the
first import must be ``from utils import context`` rather than ``from src.utils``.
Importing it is what puts the repository root on the path for everything after.
"""

import os
import sys

import click

from utils import context


config = context.config
logger = context.logger


def task_launch_api():
    """
    Launch the API with uvicorn, replacing this process.

    Uses ``os.execv`` rather than ``subprocess.run``: as a child process uvicorn never
    receives the SIGTERM that ECS sends to PID 1 on a deploy, so the parent would exit
    while uvicorn kept serving orphaned until the stop timeout forced a kill, dropping
    in-flight requests. Replacing the image makes uvicorn PID 1, so it receives signals
    directly and drains properly.

    Invoked through ``sys.executable -m uvicorn`` rather than the ``uvicorn`` script:
    the console script is only on PATH when the environment is activated, so a bare
    ``execvp("uvicorn", ...)`` fails with ENOENT under a non-activated venv, cron, or a
    container whose PATH differs. Going through the running interpreter guarantees the
    same environment that started the CLI.

    This call does not return.

    :return: never returns; the process image is replaced
    :raises OSError: if the interpreter cannot be re-executed
    """

    command = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", str(config.api_config.host),
        "--port", str(config.api_config.port),
    ]

    logger.info(f"task_launch_api replacing process: host={config.api_config.host} "
                f"port={config.api_config.port} interpreter={sys.executable}")

    os.execv(sys.executable, command)


# The task registry. Each entry is a zero-argument callable; add scheduled work here.
TASKS = {
    'task_launch_api': task_launch_api,
}

# Pipelines run their tasks in order.
PIPELINES = {
    'pipeline_api': ['task_launch_api'],
}

# Tasks that replace the process image and therefore never return. A pipeline may only
# end with one of these -- anything listed after would silently never run.
TERMINAL_TASKS = {'task_launch_api'}


def _run_task(name: str):
    """
    Run one registered task, logging its outcome.

    :param name: key in :data:`TASKS`
    :return: None
    :raises click.ClickException: if the task raises, so the CLI exits non-zero with a
        readable message rather than a traceback
    """

    logger.info(f"task starting: task={name}")

    try:
        TASKS[name]()
    except Exception as e:
        logger.error(f"task failed: task={name} error={e}")
        raise click.ClickException(f"Task '{name}' failed: {e}") from e

    logger.info(f"task finished: task={name}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=config.repo.version, prog_name=config.repo.name)
def cli():
    """Run a veritree datascience service task or pipeline."""


@cli.command(name='task')
@click.argument('name', type=click.Choice(sorted(TASKS), case_sensitive=False))
def run_task(name):
    """Run a single named task."""

    _run_task(name)


@cli.command(name='pipeline')
@click.argument('name', type=click.Choice(sorted(PIPELINES), case_sensitive=False))
def run_pipeline(name):
    """Run every task in a named pipeline, in order."""

    tasks = PIPELINES[name]

    # A terminal task replaces the process, so anything after it would never run.
    # Catching that here turns a silently skipped task into a startup error.
    for position, task in enumerate(tasks):
        if task in TERMINAL_TASKS and position != len(tasks) - 1:
            raise click.ClickException(
                f"Pipeline '{name}' lists '{task}' at position {position + 1} of "
                f"{len(tasks)}, but it replaces the process and never returns; "
                f"it must be the final task")

    logger.info(f"pipeline starting: pipeline={name} tasks={tasks}")

    for task in tasks:
        _run_task(task)

    logger.info(f"pipeline finished: pipeline={name}")


@cli.command(name='list')
def list_registry():
    """List the available tasks and pipelines."""

    click.echo("Tasks:")
    for name in sorted(TASKS):
        terminal = "  (replaces the process)" if name in TERMINAL_TASKS else ""
        click.echo(f"  {name}{terminal}")

    click.echo("\nPipelines:")
    for name, tasks in sorted(PIPELINES.items()):
        click.echo(f"  {name}: {' -> '.join(tasks)}")


def main():
    """
    CLI entry point.

    :return: the process exit status
    """

    logger.info(f"service starting: name={config.repo.name} "
                f"version={config.repo.version} argv={' '.join(sys.argv[1:])}")

    return cli(standalone_mode=True)


if __name__ == '__main__':
    main()
