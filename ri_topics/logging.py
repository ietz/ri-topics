import logging
import warnings

import numba
from loguru import logger


def setup_logging():
    hide_known_warnings()
    intercept_standard_logging()


def hide_known_warnings():
    warnings.filterwarnings('ignore', category=numba.errors.NumbaPerformanceWarning)


def intercept_standard_logging():
    logging.basicConfig(level=0, handlers=[InterceptHandler()])


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            logger.level(record.levelname)
            level = record.levelname
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
