"""
utils.py
Shared utility functions used across the project.
"""

import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "results/figures",
    "results/reports",
    "notebooks",
]


def ensure_dirs():
    """Create all required project directories if they don't exist."""
    for d in REQUIRED_DIRS:
        path = os.path.join(BASE_DIR, d)
        os.makedirs(path, exist_ok=True)
    print("[INFO] Project directories verified.")


def log(message: str):
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def get_path(*parts: str) -> str:
    """Return an absolute path relative to the project root."""
    return os.path.join(BASE_DIR, *parts)
