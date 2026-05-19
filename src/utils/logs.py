"""Tiny logging helper — consistent format across all modules."""
from __future__ import annotations
import logging
import os
import sys


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Set root logger format once; safe to call multiple times."""
    level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        root.addHandler(h)
    root.setLevel(level)
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging(os.environ.get("DREAM_LOG", "INFO"))
    return logging.getLogger(name)
