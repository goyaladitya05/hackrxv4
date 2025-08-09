import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = "http_requests.log"

os.makedirs(LOG_DIR, exist_ok=True)  # Create logs folder if not exists

logger = logging.getLogger("http_logger")
logger.setLevel(logging.INFO)

# Rotate log file after it reaches 5MB, keep 3 backups
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, LOG_FILE),
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding="utf-8"
)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)
