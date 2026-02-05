import logging
import sys

class MyLogger:
    def __init__(self, log_file="logger.log"):
        # Use the global logger as base
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers

        if not self.logger.handlers:
            # Add separate file handler
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

            # Optionally also add console output
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(ch)

    def log(self, message="base log message"):
        self.logger.info(message)

    def error_log(self, error="base log error"):
        self.logger.error(error)   