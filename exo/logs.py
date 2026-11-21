"""Run logs in the style of the Open-Source Leg logger (opensourceleg.logging), without depending on it.

    from exo.logs import LOGGER

    LOGGER.info("homed")                         # [2026-11-10 22:55:01,123] INFO: homed  (terminal and .log)
    LOGGER.track_function(lambda: x, "x")        # a CSV column (or lists of functions and of names)
    LOGGER.update()                              # one CSV row: every tracked function, in order
    LOGGER.flush_buffer()                        # write the buffered rows (between moves)
    LOGGER.close()                               # flush and close the CSV

As in OSL: one shared Logger (a logging.Logger subclass, the singleton LOGGER); messages formatted
"[time] LEVEL: message" on the terminal (INFO and up) and in <name>.log (DEBUG and up); tracked
values in <name>.csv next to it, buffered and written in blocks. Joint sets it up for every run:
logs/<log_name>_<date>_<time>.log / .csv.

Differences from OSL, on purpose:
  - Messages go to stdout (OSL: stderr), the stream print() uses, so prompts and messages stay in order.
  - The buffer holds 100000 rows by default (~500 s at 200 Hz; OSL: 1000) and Joint flushes it between
    moves: writing to the SD card inside the control loop can stall it for several milliseconds.
  - Floats are written with 6 significant digits and nan as an empty cell (OSL: str(value)).
  - A tracked function that raises writes ERROR in its cell and warns once; it stays tracked (OSL
    untracks it after 5 errors, which shifts the columns of every later row).
  - set_log_path() sets the folder (OSL has no public setter).
"""

import csv
import logging
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

ROOT = Path(__file__).resolve().parent.parent  # repository root
LOG_DIR = ROOT / "logs"

LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"  # OSL's default
BUFFER_SIZE = 100000

__all__ = ["LOGGER", "LOG_DIR", "LogLevel", "Logger", "ROOT", "write_csv"]


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger(logging.Logger):
    """Messages to the terminal and <log_path>/<file_name>.log, tracked values to <file_name>.csv.
    A singleton: every Logger() is the same object as LOGGER."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_path: Union[str, Path] = LOG_DIR, log_format: str = LOG_FORMAT,
                 file_level: LogLevel = LogLevel.DEBUG, stream_level: LogLevel = LogLevel.INFO,
                 buffer_size: int = BUFFER_SIZE) -> None:
        with self._lock:
            if getattr(self, "_initialized", False):  # the singleton exists already: keep its settings
                return
            super().__init__("exo")
            self.setLevel(logging.DEBUG)  # the handlers filter
            self._log_path = Path(log_path)
            self._formatter = logging.Formatter(log_format)
            self._file_level = file_level
            self._stream_handler = logging.StreamHandler(sys.stdout)
            self._stream_handler.setLevel(stream_level.value)
            self._stream_handler.setFormatter(self._formatter)
            self.addHandler(self._stream_handler)
            self._file_handler: Optional[logging.FileHandler] = None
            self._file_path: Optional[Path] = None
            self._csv_path: Optional[Path] = None
            self._funcs: list[Callable[[], Any]] = []
            self._names: list[str] = []
            self._failed: set[int] = set()  # tracked functions that raised (warned once)
            self._buffer: list[list] = []
            self._buffer_size = buffer_size
            self._file = None  # the open CSV file
            self._writer = None
            self._initialized = True

    # --------------------------------------------------------------------------------------------
    # Files and settings
    # --------------------------------------------------------------------------------------------
    def set_log_path(self, log_path: Union[str, Path]) -> None:
        """Folder of the .log / .csv files (takes effect at the next set_file_name)."""
        with self._lock:
            self._log_path = Path(log_path)

    def set_file_name(self, file_name: str) -> None:
        """Start writing <log_path>/<file_name>.log and .csv (an extension in file_name is dropped).
        A CSV still open is flushed and closed first."""
        with self._lock:
            self.close()
            self._log_path.mkdir(parents=True, exist_ok=True)
            stem = self._log_path / Path(file_name).stem
            self._file_path, self._csv_path = stem.with_suffix(".log"), stem.with_suffix(".csv")
            self._remove_file_handler()
            self._file_handler = logging.FileHandler(self._file_path, mode="w", encoding="utf-8")
            self._file_handler.setLevel(self._file_level.value)
            self._file_handler.setFormatter(self._formatter)
            self.addHandler(self._file_handler)

    def set_stream_level(self, level: LogLevel) -> None:
        self._stream_handler.setLevel(level.value)

    def set_file_level(self, level: LogLevel) -> None:
        self._file_level = level
        if self._file_handler is not None:
            self._file_handler.setLevel(level.value)

    def set_format(self, log_format: str) -> None:
        self._formatter = logging.Formatter(log_format)
        for handler in self.handlers:
            handler.setFormatter(self._formatter)

    def set_stream_terminator(self, terminator: str) -> None:
        self._stream_handler.terminator = terminator

    def set_buffer_size(self, buffer_size: int) -> None:
        """Rows kept in memory before update() writes them (flush_buffer() writes them any time)."""
        if buffer_size <= 0:
            self.warning(f"Invalid buffer size: {buffer_size}. Using default of {BUFFER_SIZE}.")
            buffer_size = BUFFER_SIZE
        self._buffer_size = buffer_size

    # --------------------------------------------------------------------------------------------
    # Tracked values (CSV)
    # --------------------------------------------------------------------------------------------
    def track_function(self, var_func: Union[Callable[[], Any], list[Callable[[], Any]]],
                       name: Union[str, list[str]]) -> None:
        """Add CSV columns: a function and its name, or a list of functions and a list of names.
        Columns are written in the order they were added; add them all before the first update()."""
        with self._lock:
            if isinstance(var_func, list) and isinstance(name, list):
                if len(var_func) != len(name):
                    raise ValueError("The number of variable functions and names must match.")
                funcs, names = var_func, name
            elif callable(var_func) and isinstance(name, str):
                funcs, names = [var_func], [name]
            else:
                raise TypeError("var_func and name must both be single values or lists of equal length.")
            if self._file is not None:
                raise RuntimeError("columns cannot be added once the CSV header is written")
            self._funcs.extend(funcs)
            self._names.extend(names)
            for n in names:
                self.debug(f"Started tracking variable: {n}")

    def track_attributes(self, obj: Any, attributes: Union[str, list[str]]) -> None:
        """Add CSV columns for attributes of obj (named after the attributes)."""
        attributes = [attributes] if isinstance(attributes, str) else attributes
        self.track_function([lambda a=a: getattr(obj, a) for a in attributes], list(attributes))

    def get_tracked_variables(self) -> list[tuple[str, Any]]:
        """(name, current value) of every tracked column."""
        result = []
        for n, f in zip(self._names, self._funcs):
            try:
                result.append((n, f()))
            except Exception as e:
                result.append((n, f"ERROR: {e}"))
        return result

    def update(self) -> None:
        """Buffer one CSV row: the current value of every tracked function."""
        if not self._funcs:
            return
        with self._lock:
            row = []
            for i, f in enumerate(self._funcs):
                try:
                    row.append(_format(f()))
                except Exception as e:
                    row.append("ERROR")
                    if i not in self._failed:
                        self._failed.add(i)
                        self.warning(f"Error getting value for {self._names[i]}: {e!r} (logged as ERROR)")
            self._buffer.append(row)
            if len(self._buffer) >= self._buffer_size:
                self.flush_buffer()

    def flush_buffer(self) -> None:
        """Write the buffered rows to the CSV (opened, with its header, on the first write)."""
        with self._lock:
            if not self._buffer or not self._open_csv():
                return
            self._writer.writerows(self._buffer)
            self._buffer.clear()
            self._file.flush()

    def close(self) -> None:
        """Flush the buffered rows and close the CSV (written with its header even without rows).
        The .log file stays open: messages after close() still go to it."""
        with self._lock:
            if self._funcs:
                self._open_csv()
            self.flush_buffer()
            if self._file is not None:
                self._file.close()
                self._file, self._writer = None, None

    def reset(self) -> None:
        """Close the files and forget the tracked columns (before setting up a new run)."""
        with self._lock:
            self.close()
            self._remove_file_handler()
            self._funcs.clear()
            self._names.clear()
            self._failed.clear()
            self._buffer.clear()
            self._file_path = self._csv_path = None

    # --------------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------------
    @property
    def file_path(self) -> Optional[Path]:
        """The .log file (None until set_file_name)."""
        return self._file_path

    @property
    def csv_path(self) -> Optional[Path]:
        """The .csv file (None until set_file_name)."""
        return self._csv_path

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def tracked_variable_count(self) -> int:
        return len(self._funcs)

    def __repr__(self) -> str:
        return f"Logger(file_path={self._file_path}, tracked_vars={len(self._funcs)})"

    # --------------------------------------------------------------------------------------------
    def _open_csv(self) -> bool:
        if self._file is None:
            if self._csv_path is None:  # no set_file_name: messages only, rows are dropped
                self._buffer.clear()
                return False
            self._file = open(self._csv_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._names)
        return True

    def _remove_file_handler(self) -> None:
        if self._file_handler is not None:
            self.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None


LOGGER = Logger()


def _format(value: object) -> object:
    if isinstance(value, float):
        return "" if value != value else f"{value:.6g}"  # nan -> empty cell
    return value


def write_csv(path: Path, columns: list[str], rows: list) -> None:
    """Write rows (sequences, in the order of `columns`) to a CSV file (for result files)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows([[_format(v) for v in row] for row in rows])
