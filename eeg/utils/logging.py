# utils/logging_utils.py
import sys
import logging
from pathlib import Path


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler(sys.__stdout__)]
    )

    # ---------- Tee stdout / stderr into the logger ----------
    class _LoggerWriter:
        def __init__(self, _logger, _level):
            self._logger = _logger
            self._level = _level

        def write(self, message):
            msg = message.rstrip()
            if msg: self._logger.log(self._level, msg)

        def flush(self): pass

    root_logger = logging.getLogger()
    sys.stdout = _LoggerWriter(root_logger, logging.INFO)
    sys.stderr = _LoggerWriter(root_logger, logging.ERROR)
