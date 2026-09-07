"""
Command-line entry point for the veritree datascience service.

Dispatches named tasks and pipelines, the convention shared across the veritree
datascience repositories and the contract the ECS task definitions invoke:

    python3 src/main.py pipeline pipeline_api
    python3 src/main.py task task_launch_api

Today the only task launches the API; scheduled work belongs in the same registry.
"""

import os
import subprocess

import click
import datetime

from utils import context


config = context.config
logger = context.logger


datenow = datetime.datetime.now()
logger.info(f'VERITREE DATASCIENCE SERVICE STARTED: {datenow}')


def call_task_launch_api():
    """ Launch the unified datascience FastAPI service (computer_vision + datascience_results) """
    subprocess.run([
        "uvicorn", "api.main:app",
        "--host", str(config.api_config.host),
        "--port", str(config.api_config.port),
    ])
    return None


task_names = {
    'task_launch_api': call_task_launch_api,
}

pipeline_names = {
    'pipeline_api': ['task_launch_api'],
}


@click.command()
@click.argument('run_type', required=1)
@click.argument('run_arg', required=1)
def run_service(run_type, run_arg, task_dict=task_names, pipeline_dict=pipeline_names):
    """
    Run a named task, or every task in a named pipeline.

    :param run_type: either 'task' or 'pipeline'
    :param run_arg: name of the task or pipeline to run
    :param task_dict: mapping of task name to the function implementing it
    :param pipeline_dict: mapping of pipeline name to its ordered list of task names
    :return: None
    """
    print(os.getcwd())

    if run_type == 'pipeline':
        tasks_to_run = pipeline_dict[run_arg]

        for i_tasks_to_run in tasks_to_run:
            task_function = task_dict[i_tasks_to_run]
            task_function.__call__()

    elif run_type == 'task':
        task_function = task_dict[run_arg]
        task_function.__call__()

    return None


if __name__ == '__main__':
    run_service()

# Run pipeline_main using pipeline or Run task_x using task
# > python3 ./src/main.py pipeline pipeline_api
