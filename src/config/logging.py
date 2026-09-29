"""Run-log setup."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from src.config.paths import BASE_DIR


def setup_logging(name: str, logs_dir: Path | None = None) -> logging.Logger:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (logs_dir or BASE_DIR / "logs") / f"run_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)
