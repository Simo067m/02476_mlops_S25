from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add("logs/debug.log", level="DEBUG", rotation="10 MB")
logger.add("logs/warning.log", level="WARNING", rotation="10 MB")

log = logger

if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
