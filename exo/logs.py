"""Run logs: messages to the terminal and a .log file, and one CSV row per control tick.

Every run writes to logs/<name>_<date>_<time>.log and .csv (Joint sets this up).
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent  # repository root
LOG_DIR = ROOT / "logs"

log = logging.getLogger("exo")


def start_run_log(name: str) -> Path:
    """Send log messages to the terminal (info and up) and to logs/<name>_<time>.log (everything).
    Returns the path without extension; the run's other files are named after it."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    n = 2
    while stem.with_suffix(".log").exists():  # two runs within the same second
        stem = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}_{n}"
        n += 1
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for handler in list(log.handlers):  # a previous run in the same Python session
        log.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)  # same stream as print(), so the order is kept
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(f"{stem}.log")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(console)
    log.addHandler(file)
    return stem


class TickLog:
    """CSV file with one row per control tick. `columns` maps each column name to a function that
    returns its value. Rows are kept in memory and written in blocks (flush()), because writing to
    the SD card inside the control loop can stall it for several milliseconds."""

    def __init__(self, path: Path, columns: dict[str, Callable[[], object]], max_buffer: int = 20000) -> None:
        self.path = path
        self._names, self._getters = list(columns), list(columns.values())
        self._rows: list[list] = []
        self._max_buffer = max_buffer
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._names)

    def add_row(self) -> None:
        self._rows.append([_format(get()) for get in self._getters])
        if len(self._rows) >= self._max_buffer:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows (call between moves, not inside a fast loop)."""
        if self._rows:
            self._writer.writerows(self._rows)
            self._rows.clear()
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self.flush()
            self._file.close()


def _format(value: object) -> object:
    if isinstance(value, float):
        return "" if value != value else f"{value:.6g}"  # nan -> empty cell
    return value


def write_csv(path: Path, columns: list[str], rows: list) -> None:
    """Write rows (sequences, in the order of `columns`) to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows([[_format(v) for v in row] for row in rows])
