"""
Project-wide configuration and logging.

Importing this module loads the .env file, merges every configs/*.yaml into a single
flat namespace, and configures the shared logger. Both are exposed as module-level
singletons that the rest of the service imports:

    from src.utils import context
    config = context.config
    logger = context.logger
"""

import logging
import logging.handlers
import os
import sys
import yaml

import dotenv
from box import Box


def startup():
    """
    Initialize all necessary directories and files and perform checks on overall system health and all needed params
    :return:
    """

    dotenv.load_dotenv()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
    sys.path.insert(0, ROOT_DIR)
    return ROOT_DIR

root_dir = startup()


def load_config(config_main_path: str = os.path.join(root_dir, 'configs')) -> Box:
    """
    Create all the configuration files and make available throughout project via context.py

    Unlike the single-config loader inherited from the source repositories, this service
    splits configuration across several yaml files (config.yaml, one config_cv_*.yaml
    per computer-vision capability, and config_datascience_results.yaml). Every file is
    merged into ONE flat namespace, so
    callers keep the familiar `config.<key>` access pattern regardless of which file a key
    lives in. A key defined in more than one file is a configuration error and raises.

    :param config_main_path: directory containing config files
    :return: Box object containing config variables
    """

    config_files = sorted(
        i_config_file for i_config_file in os.listdir(config_main_path)
        if i_config_file.endswith('.yaml')
    )

    config_merged = {}

    for config_file_name in config_files:
        i_config_file_path = os.path.join(config_main_path, config_file_name)

        with open(i_config_file_path, 'r') as yaml_file:
            logging.info(f"Loading configuration file: {config_file_name}")
            i_config = yaml.safe_load(yaml_file) or {}

        collisions = set(i_config).intersection(config_merged)
        if collisions:
            raise ValueError(
                f"Duplicate configuration keys {sorted(collisions)} found in {config_file_name}; "
                f"each top-level key must be defined in exactly one configs/*.yaml file"
            )

        config_merged.update(i_config)

    return Box(config_merged)


def load_logging(logging_main_path: str = os.path.join(root_dir, 'logs')):
    """
    Configure the project-wide logger.

    Writes to a size-rotated file plus stdout. Rotation matters in the deployed
    container: without it the log grows unbounded until it fills the task's
    ephemeral storage. Defaults give at most
    ``(max_bytes * (backup_count + 1))`` on disk, and are overridable under
    ``logging:`` in configs/config.yaml.

    stdout is retained alongside the file because ECS ships the stream to
    CloudWatch; the file is for exec-ing into a running task.

    :param logging_main_path: directory to hold the log file, created if absent
    :return: the configured ``logging.Logger``
    """

    logger = logging.getLogger("logger")

    # Idempotency guard: loggers are process-global singletons keyed by name, so
    # getLogger("logger") returns the same object on every call. This module can be
    # imported under two identities in one run -- `utils.context` (via src/main.py,
    # where sys.path[0] is src/) and `src.utils.context` (via the libs/routes) --
    # each of which executes load_logging(). Without this guard the second import
    # stacks another stream + file handler onto the same logger, so every line is
    # emitted twice. Only attach handlers once.
    if logger.handlers:
        return logger

    # config is not loaded yet at this point in module execution, so the rotation
    # settings are read directly rather than through the Box.
    settings = _logging_settings()

    os.makedirs(logging_main_path, exist_ok=True)
    log_full_path = os.path.join(logging_main_path, settings['filename'])

    logging.root.handlers = []

    logger.setLevel(getattr(logging, settings['level'], logging.INFO))
    logger.propagate = False

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s: %(funcName)s(): %(lineno)s]\n  ---  %(message)s\n"
    )

    log_filehandler = logging.handlers.RotatingFileHandler(
        log_full_path,
        maxBytes=settings['max_bytes'],
        backupCount=settings['backup_count'],
        encoding='utf-8',
        delay=True,
    )
    log_filehandler.setFormatter(log_formatter)
    logger.addHandler(log_filehandler)

    log_streamhandler = logging.StreamHandler(sys.stdout)
    log_streamhandler.setFormatter(log_formatter)
    logger.addHandler(log_streamhandler)

    return logger


def _logging_settings() -> dict:
    """
    Read the ``logging:`` block out of configs/config.yaml, falling back to defaults.

    Read directly from the yaml rather than via load_config(), because logging is
    configured before the merged config Box exists.

    :return: dict with filename, level, max_bytes and backup_count
    """

    defaults = {
        'filename': 'veritree_datascience_service.log',
        'level': 'INFO',
        'max_bytes': 10 * 1024 * 1024,   # 10 MB per file
        'backup_count': 5,               # plus 5 rotations == 60 MB ceiling
    }

    try:
        with open(os.path.join(root_dir, 'configs', 'config.yaml'), 'r') as yaml_file:
            block = (yaml.safe_load(yaml_file) or {}).get('logging') or {}
    except (OSError, yaml.YAMLError):
        return defaults

    return {**defaults, **{k: v for k, v in block.items() if k in defaults}}


logger = load_logging()
config = load_config()
