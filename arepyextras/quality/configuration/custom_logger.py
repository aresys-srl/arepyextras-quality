# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Custom logger setup"""

import logging
from enum import Enum


class AnsiColors(Enum):
    """Ansi escape color strings for Logging Formatter"""

    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    GREEN = "\x1b[1;32m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    PURPLE = "\x1b[1;35m"
    BLUE = "\x1b[1;34m"
    LIGHT_BLUE = "\x1b[1;36m"
    RESET = "\x1b[0m"


class CustomFormatter(logging.Formatter):
    """Custom logger formatter with colors"""

    # message formatting layout
    fmt = "| %(levelname)-9s @ %(module)s| %(asctime)s | %(message)s"

    FORMATS = {
        logging.DEBUG: AnsiColors.GREY.value + fmt + AnsiColors.RESET.value,
        logging.INFO: AnsiColors.GREY.value + fmt + AnsiColors.RESET.value,
        logging.WARNING: AnsiColors.YELLOW.value + fmt + AnsiColors.RESET.value,
        logging.ERROR: AnsiColors.RED.value + fmt + AnsiColors.RESET.value,
        logging.CRITICAL: AnsiColors.BOLD_RED.value + fmt + AnsiColors.RESET.value,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


class MyHandler(logging.StreamHandler):
    """Custom logging stream handler to centralize logging"""

    def __init__(self):
        logging.StreamHandler.__init__(self)
        self.setFormatter(CustomFormatter())
