"""
Logging manager for the multi-agent support system.
Handles all logging configuration and setup.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .config import Config


class LoggingManager:
    """Manages logging configuration and setup for the system."""

    def __init__(self, config: Config):
        """Initialize the logging manager."""
        self.config = config
        self._loggers: Dict[str, logging.Logger] = {}

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self._loggers:
            self._loggers[name] = self._create_logger(name)
        return self._loggers[name]

    def _create_logger(self, name: str) -> logging.Logger:
        """Create a new logger with proper configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.config.logging.level))

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Create logs directory if it doesn't exist
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # File handler - always add this
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.logging.level))

        # Console handler - only add if not in interactive mode
        # This prevents log spam in the Rich UI
        import sys
        if len(sys.argv) > 1 or '--verbose' in sys.argv:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # Only show warnings and errors on console

            # Custom formatter for console (shorter format)
            console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Formatter for file
        file_formatter = logging.Formatter(self.config.logging.format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def update_log_level(self, level: str) -> None:
        """Update log level for all existing loggers."""
        log_level = getattr(logging, level.upper())
        for logger in self._loggers.values():
            logger.setLevel(log_level)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(log_level)

    def get_system_info(self) -> Dict[str, Any]:
        """Get logging system information."""
        return {
            "log_level": self.config.logging.level,
            "log_file": self.config.logging.file,
            "active_loggers": list(self._loggers.keys())
        }
