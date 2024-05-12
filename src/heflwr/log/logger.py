import sys
import logging
from logging.handlers import HTTPHandler
from logging import LogRecord
from typing import Any, Dict, Optional, Tuple, Union
from ssl import SSLContext
import warnings

# Create logger
LOGGER_NAME = "hetero-flwr"
HETERO_FLOWER_LOGGER = logging.getLogger(LOGGER_NAME)
HETERO_FLOWER_LOGGER.setLevel(logging.DEBUG)

# Console default formatter
COMMON_LOG_FORMAT = "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
DEFAULT_FORMATTER = logging.Formatter(COMMON_LOG_FORMAT)

# Configure console handler
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(DEFAULT_FORMATTER)
HETERO_FLOWER_LOGGER.addHandler(console_handler)


# Modify HTTPHandler, Overwrite mapLogRecord, Add `identifier`
class CustomHTTPHandler(HTTPHandler):
    def __init__(
        self,
        identifier: str,
        host: str,
        url: str,
        method: str = "GET",
        secure: bool = False,
        credentials: Optional[Tuple[str, str]] = None,
        context: Optional[Union[None, SSLContext]] = None
    ) -> None:
        super().__init__(host, url, method, secure, credentials, context)
        self.identifier = identifier

    def mapLogRecord(self, record: LogRecord) -> Dict[str, Any]:
        return {"identifier": self.identifier, **record.__dict__}


# SimpleHTTPHandler reduce the network cost
class SimpleHTTPHandler(CustomHTTPHandler):
    def mapLogRecord(self, record: LogRecord) -> Dict[str, Any]:
        return {"identifier": self.identifier, "message": record.__dict__["message"]}


# Add file handler or/and http handler
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


# export
logger = logging.getLogger(LOGGER_NAME)
log = logger.log
configure = configure


# if __name__ == '__main__':
#     configure("temp-identifier")
#     logger.info("hello world")
#
#     configure("temp-identifier", file="./test_logger_identifier.txt")
#     logger.info("hello world")
#
#     configure("temp-identifier", host="127.0.0.1:5000")
#     logger.info("hello world")
#
#     configure("temp-identifier", file="./test_logger_identifier.txt", host="127.0.0.1:5000", simple=True)
#     logger.info("hello world")

