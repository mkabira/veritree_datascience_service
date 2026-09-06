import logging
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
    splits configuration across several yaml files (config.yaml, config_computer_vision.yaml,
    config_datascience_results.yaml). Every file is merged into ONE flat namespace, so
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
    Setup logging systems accessible for entire project
    :param logging_main_path:
    :return:
    """

    os.makedirs(logging_main_path, exist_ok=True)

    log_full_path = os.path.join(logging_main_path, 'veritree_datascience_service.log')

    if not os.path.exists(log_full_path):
        f = open(log_full_path, "x")
        f.close()

    logging.root.handlers = []

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Idempotency guard: loggers are process-global singletons keyed by name,
    # so getLogger("logger") returns the same object on every call. This module
    # can be imported under two identities in one run -- `utils.context` (via
    # src/main.py, where sys.path[0] is src/) and `src.utils.context` (via the
    # libs/tasks) -- each of which executes load_logging(). Without this guard
    # the second import stacks another stream + file handler onto the same
    # logger, so every line is emitted twice. Only attach handlers once.
    if logger.handlers:
        return logger

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s: %(funcName)s(): %(lineno)s]\n  ---  %(message)s\n"
    )

    log_filehandler = logging.FileHandler(log_full_path, mode='a')
    log_filehandler.setFormatter(log_formatter)
    logger.addHandler(log_filehandler)

    log_streamhandler = logging.StreamHandler(sys.stdout)
    log_streamhandler.setFormatter(log_formatter)
    logger.addHandler(log_streamhandler)

    return logger


logger = load_logging()
config = load_config()
