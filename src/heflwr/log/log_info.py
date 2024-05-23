import sys
import logging

# Create logger
LOGGER_NAME = "HeFlwr"
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

# Export
logger = logging.getLogger(LOGGER_NAME)
log = logger.log
