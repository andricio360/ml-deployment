"""
IrisLogger class to log messages to the console and to a file.
"""

import sys

from loguru import logger


class IrisLogger:
    """
    IrisLogger class to log messages to the console and to a file.

    Attributes:
    - logger: loguru.logger object

    Methods:
    - configure_logger: Configures the logger to log messages to the console and to a file.
    - debug: Logs a debug message.
    - info: Logs an info message.
    - warning: Logs a warning message.
    - error: Logs an error message.
    - critical: Logs a critical message.
    - add: Adds a new sink to the logger.
    - remove: Removes all sinks from the logger.
    """

    def __init__(self, logger: logger) -> None:
        """
        Constructor method to initialize the IrisLogger class.
        """
        self._logger = logger
        self.configure_logger()

    def configure_logger(self) -> None:
        """
        Configures the logger to log messages to the console and to a file.
        """
        self._logger.remove()
        self._logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
        self._logger.add("logs/iris_logger.log", level="INFO")

    def debug(self, message) -> None:
        """
        Logs a debug message.

        Args:
        - message: str
        """
        self._logger.debug(message)

    def info(self, message) -> None:
        """
        Logs an info message.
        Args:
        - message: str
        """
        self._logger.info(message)

    def warning(self, message) -> None:
        """
        Logs a warning message.
        Args:
        - message: str
        """
        self._logger.warning(message)

    def error(self, message) -> None:
        """
        Logs an error message.
        Args:
        - message: str
        """
        self._logger.error(message)

    def critical(self, message) -> None:
        """
        Logs a critical message.
        Args:
        - message: str
        """
        self._logger.critical(message)

    def add(self, sink, **kwargs) -> None:
        """
        Adds a new sink to the logger.
        Args:
        - sink: str
        - kwargs: dict
        """
        self._logger.add(sink, **kwargs)

    def remove(self) -> None:
        """
        Removes all sinks from the logger.
        """
        self._logger.remove()
