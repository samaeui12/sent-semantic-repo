import os
import logging
import sys

class Experi_Logger:
    def __init__(self, output_dir='temp', level='info'):
        self.output_dir = output_dir
        self.level = level
        self._reset_logging()
        
    def _reset_logging(self):
        """
        Reset and configure logging settings.
        """
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        root_logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, 'execute.log')

        if self.level =='info':
            self.level = logging.INFO
        elif self.level =='error':
            self.level = logging.ERROR
        elif self.level =='warning':
            self.level = logging.WARNING

        logging.basicConfig(filename=log_path, level=self.level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)