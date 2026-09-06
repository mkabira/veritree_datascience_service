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
    run_type (str): one of ['task' or 'pipeline']
    run_arg (str): name of task or pipeline to run
    task_dict (dict): dictionary of task names and task function mappings
    pipeline_dict (dict): dictionary of pipeline names and task function mappings
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
