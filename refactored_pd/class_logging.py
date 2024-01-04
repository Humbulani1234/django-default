import logging
import re


class DiagnosticLogger:
    def __init__(self, logger_name, log_level=logging.DEBUG):
        self.diagnostic_logger = logging.getLogger(logger_name)
        self.diagnostic_logger.setLevel(log_level)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(
            logging.Formatter(fmt="{levelname}:{name}:{message}", style="{")
        )
        self.diagnostics_logger.addHandler(console_handler)

    def __str__(self):
        pattern = re.compile(r"^_")
        method_names = []
        for name, func in QuantileResiduals.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def log_info(self, message):
        return self.diagnostics_logger.info(message)

    def log_error(self, message):
        return self.diagnostics_logger.error(message)
