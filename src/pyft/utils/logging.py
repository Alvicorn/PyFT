import logging
from logging import Logger
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import Optional


class CustomFormatter(logging.Formatter):
    """
    Custom format for the application logger
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[38;5;39m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def serial_logger(
    log_file: Optional[str] = None, log_level=logging.DEBUG
) -> Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(CustomFormatter())
        logger.addHandler(fh)

    return logger


"""
    concurrent logger use:

    1. create a threading/multiprocessing queue object
    2. define a common log file name
    3. create the queue handler with concurrent_logger_handler.
       each process needs to define their own logger, and add the queue handler
    4. create a listener process with concurrent_logger_listener. when ready,
       start the process. don't forget to stop the process at the end.

"""


def concurrent_logger_handler(log_queue: Queue, log_file: str) -> QueueHandler:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create a file handler for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(CustomFormatter())

    # add handler to the logger
    logger.addHandler(file_handler)

    # create a queue handler to send logs from worker processes
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    return queue_handler


def concurrent_logger_listener(
    log_queue: Queue, log_file: str
) -> QueueListener:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create a file handler for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # create and set formatter
    formatter = CustomFormatter()
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return QueueListener(log_queue, file_handler)
