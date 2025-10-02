# src/serving/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

DEFAULT_LEVEL = logging.INFO


def configure_root_logger(level: int = DEFAULT_LEVEL):
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s'
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str):
    configure_root_logger()
    return logging.getLogger(name)
