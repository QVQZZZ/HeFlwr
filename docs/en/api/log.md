# heflwr.log

`heflwr.log` module is responsible for logging and tracing the running status of the entire project. This module provides three components: `logger`, `log`, and `configure`.

## logger
`logger` is a `logging.Logger` object with the name `hetero-flwr`. It is a global logger (_per-project logger_) and can be used like a common `logging.Logger` object. For example:

```python
from heflwr.log import logger

logger.debug("hello, world!")
logger.info("hello, world!")
logger.warning("hello, world!")
logger.error("hello, world!")
logger.critical("hello, world!")
```
The `logger` object is by default bound to `logging.StreamHandler(stream=sys.stdout)`, so the above example will print logs containing `"hello, world!"` to the standard output.

## log
The `log` method of the logger provides a quick way to log messages directly. It is equivalent to the global logger's `log` method. You only need to pass the log level and the message, for example:
```python
from logging import INFO
from heflwr.log import log

log(INFO, "hello, world!")
```
The example above will print a log with the message `"hello, world!"` to the standard output.

## configure

The `configure` function is used to customize the behavior of the global logger, such as specifying log identifiers, output file paths, and remote HTTP server addresses. It accepts the following parameters:
- `identifier: str`: A string identifier that will be added as a prefix to each log entry to help trace the source of the log. A typical use is to set it as a client identifier, such as "client-A", to indicate the source of the log to a remote server.
- `file: Optional[str] = None`: An optional parameter for specifying the path to the log output file. If provided, logs will be written to this file as well.
- `host: Optional[str] = None`: An optional parameter for specifying the address of a remote HTTP server. If provided, logs will be sent to this server.
- `simple: Optional[bool] = None`: An optional parameter that specifies the format of remote logs. This parameter must be used in conjunction with the `host` parameter. If the `host` parameter is set to `None`, specifying the `simple` parameter will raise a warning. If host is not `None` and `simple` is set to `None` or `False`, logs will be sent in standard format to the `/log` URL of the remote server. If `host` is not `None` and `simple` is set to `True`, logs will be sent in a concise format to the `/simple_log` URL of the remote server to reduce network transmission overhead.

Calling the `configure` function can add `FileHandler` and/or `HTTPHandler` to the global logger, for example:
```python
from heflwr.log import logger, configure

configure(identifier="my-app", file="logs/app.log")
configure(identifier="client-A", host="127.0.0.1:5000")
configure(identifier="client-A", file="logs/app.log", host="127.0.0.1:5000")
configure(identifier="client-A", file="logs/app.log", host="127.0.0.1:5000", simple=True)

logger.info("hello, world!")
```
By doing this, logs can not only be output to the standard output but also to the corresponding file and/or HTTP server.
