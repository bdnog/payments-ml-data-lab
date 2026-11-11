"""Logging setup shared by the services."""

import logging
import sys

LOG_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure the root logger with a single key=value style stream handler.

    Safe to call more than once: existing handlers are replaced, not duplicated.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
