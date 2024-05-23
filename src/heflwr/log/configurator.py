import logging
from typing import Optional
import warnings

from .log_info import COMMON_LOG_FORMAT, HETERO_FLOWER_LOGGER
from .handlers import CustomHTTPHandler, SimpleHTTPHandler


def configure(
    identifier: str,
    file: Optional[str] = None,
    host: Optional[str] = None,
    simple: Optional[bool] = None
) -> None:
    format_string = f"{identifier} | {COMMON_LOG_FORMAT}"
    formatter = logging.Formatter(format_string)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        HETERO_FLOWER_LOGGER.addHandler(file_handler)

    if host:
        if simple is None or not simple:
            http_handler = CustomHTTPHandler(identifier=identifier, host=host, url="/log", method="POST")
        else:  # simple is True
            http_handler = SimpleHTTPHandler(identifier=identifier, host=host, url="/simple_log", method="POST")
        http_handler.setLevel(logging.DEBUG)
        # Override mapLogRecord as setFormatter has no effect on what is sending via http
        # http_handler.setFormatter(formatter)
        HETERO_FLOWER_LOGGER.addHandler(http_handler)
    elif simple is not None:  # host is None and simple is not None
        warnings.warn("Simple mode is specified without a host. No HTTP handler will be added.", UserWarning)
