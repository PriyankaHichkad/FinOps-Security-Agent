import logging
import os
import sys
from datetime import datetime

# Define log directory and file
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FinOps-Security-Agent")

class FinGuardException(Exception):
    """Custom exception class with file and line number context."""
    def __init__(self, error_message, error_detail: sys = None):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error, error_detail: sys):
        if error_detail and hasattr(error_detail, "exc_info"):
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                return f"Error occurred in script [{file_name}] line [{line_number}] message [{str(error)}]"
        return str(error)

    def __str__(self):
        return self.error_message
