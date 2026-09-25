"""Tee stdout/stderr to a case-directory log file (rank 0 only)."""

from __future__ import annotations

import os
import sys
from typing import IO, Optional, TextIO

import yaml

_LOG_FILE: Optional[TextIO] = None
_STDOUT: Optional[TextIO] = None
_STDERR: Optional[TextIO] = None


class _Tee(TextIO):
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty() if self._streams else False

    def fileno(self) -> int:
        for stream in self._streams:
            fileno = getattr(stream, "fileno", None)
            if fileno is None:
                continue
            try:
                fd = fileno()
            except Exception:
                continue
            if fd is not None:
                return int(fd)
        raise OSError("_Tee has no underlying file descriptor")

    @property
    def encoding(self) -> str:
        return self._streams[0].encoding if self._streams else "utf-8"


def setup_run_log(log_path: str, *, rank: int = 0, enabled: bool = True) -> str | None:
    """Mirror rank-0 stdout/stderr to ``log_path``. Returns path when enabled."""
    global _LOG_FILE, _STDOUT, _STDERR
    if not enabled or rank != 0:
        return None
    if _LOG_FILE is not None:
        return log_path

    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    _STDOUT = sys.stdout
    _STDERR = sys.stderr
    _LOG_FILE = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_STDOUT, _LOG_FILE)
    sys.stderr = _Tee(_STDERR, _LOG_FILE)
    return log_path


def close_run_log() -> None:
    global _LOG_FILE, _STDOUT, _STDERR
    if _LOG_FILE is None:
        return
    try:
        _LOG_FILE.flush()
        _LOG_FILE.close()
    finally:
        if _STDOUT is not None:
            sys.stdout = _STDOUT
        if _STDERR is not None:
            sys.stderr = _STDERR
        _LOG_FILE = None
        _STDOUT = None
        _STDERR = None


def log_run_banner(*, argv: list[str], cfg_dict: dict, case_dir: str, log_path: str | None) -> None:
    print("=== pdebench run ===", flush=True)
    if log_path is not None:
        print(f"log_file: {log_path}", flush=True)
    print(f"case_dir: {case_dir}", flush=True)
    log_config_trace(argv=argv, cfg_dict=cfg_dict, title="CLI argv")


def log_config_trace(*, argv: list[str], cfg_dict: dict, title: str = "config") -> None:
    print(f"\n=== {title} ===", flush=True)
    print("argv:", " ".join(argv), flush=True)
    yaml.safe_dump(cfg_dict, sys.stdout, sort_keys=False)
    print(flush=True)
