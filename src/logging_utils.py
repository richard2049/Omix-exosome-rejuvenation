"""
logging_utils.py

Wrapper around the standard logging module that provides a
consistent logger configuration for all pipeline modules.
"""

import logging


def get_logger(name: str = "src_exo_pipeline") -> logging.Logger:
    """
    Create a consistent logger for pipeline modules.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt=fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger