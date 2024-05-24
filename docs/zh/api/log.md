# heflwr.log

`heflwr.log` 模块负责记录和跟踪整个项目的运行情况, 该模块提供 `logger`, `log`, `configure` 三个组件.

## logger
`logger` 是一个名称为 `hetero-flwr` 的 `logging.Logger` 对象, 它是一个全局的日志记录器
(_per-project logger_),
可以像正常的 `logging.Logger` 对象那样使用它, 例如:

```python
from heflwr.log import logger

logger.debug("hello, world!")
logger.info("hello, world!")
logger.warning("hello, world!")
logger.error("hello, world!")
logger.critical("hello, world!")
```
`logger` 对象默认绑定 `logging.StreamHandler(stream=sys.stdout)`,
因此上述示例会将携带 `"hello, world!"` 信息的日志打印到标准输出.

## log
日志记录器的 `log` 方法提供了一种直接记录日志的快捷方式, 它等价于全局日志记录器的 `log` 方法,
调用时只需要传入日志级别和日志信息, 例如:

```python
from logging import INFO
from heflwr.log import log

log(INFO, "hello, world!")
```
上述示例会将携带 `"hello, world!"` 信息的日志打印到标准输出.

## configure
`configure` 函数用于定制全局日志器的行为, 例如指定日志的标识符、输出文件路径和远程 HTTP 服务器地址等,
它接受以下参数:
- `identifier: str`: 一个标识字符串, 它将被添加到每条日志记录的前缀中, 帮助追踪日志来源.
一个典型的使用方式是将其设定为客户端标识符, 如 "client-A", 以向远程服务器标明日志的来源.
- `file: Optional[str] = None`: 一个可选参数, 用于指定日志输出文件的路径.
如果指定, 日志将被同时写入到这个文件中.
- `host: Optional[str] = None`: 一个可选参数, 用于指定远程 HTTP 服务器的地址.
如果指定, 日志将被发送到这个服务器.
- `simple: Optional[bool] = None`: 一个可选参数, 用于指定远程日志的格式.
该参数必须配合 `host` 参数使用, 若 `host` 参数被设置为 `None`, 指定 `simple` 参数将引发一个警告.
若 `host` 不为 None, 且 `simple` 被设置为 `None` 或 `False`, 则日志将以标准格式发送到远程服务器的 `/log` URL 下. 
若 `host` 不为 None, 且 `simple` 被设置为 `True`, 则日志将以简略格式发送到远程服务器的 `/simple_log` URL 下, 以降低网络传输开销.


调用 `configure` 函数可以添加`FileHandler` 和 / 或 `HTTPHandler` 到全局日志器中, 例如:

```python
from heflwr.log import logger, configure

configure(identifier="my-app", file="logs/app.log")
configure(identifier="client-A", host="127.0.0.1:5000")
configure(identifier="client-A", file="logs/app.log", host="127.0.0.1:5000")
configure(identifier="client-A", file="logs/app.log", host="127.0.0.1:5000", simple=True)

logger.info("hello, world!")
```
通过这种方式, 日志不仅可以输出到标准输出, 还会输出到对应文件和 / 或 HTTP 服务器.