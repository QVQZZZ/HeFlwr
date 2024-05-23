from logging.handlers import HTTPHandler
from logging import LogRecord
from typing import Any, Dict, Optional, Tuple, Union
from ssl import SSLContext


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
