# app/utils.py

import os
import sqlite3
from typing import Optional
from .security import is_safe_path
from .config import DATA_DIR
from .logging_conf import logger

def safe_join_data(*paths: str) -> str:
    """
    Join path segments under DATA_DIR, ensuring we never go outside.
    Logs an error if path is invalid.
    """
    full_path = os.path.join(DATA_DIR, *paths)
    if not is_safe_path(full_path):
        err_msg = f"Path {full_path} is outside /data or is not safe."
        logger.error(err_msg)
        raise PermissionError(err_msg)
    return full_path

def write_file(relative_path: str, content: str):
    """
    Write `content` to a file under /data. Overwrites existing content.
    Logs any exceptions.
    """
    try:
        full_path = safe_join_data(relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Successfully wrote file at {full_path}")
    except Exception as e:
        logger.error(f"Error writing to {relative_path}: {e}")
        raise e

def read_file(relative_path: str) -> str:
    """
    Read content from a file under /data. Logs any exceptions or if file doesn't exist.
    """
    full_path = safe_join_data(relative_path)
    if not os.path.exists(full_path):
        logger.error(f"Tried to read file {full_path}, but it does not exist.")
        raise FileNotFoundError("File does not exist.")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = f.read()
        logger.debug(f"Read file: {full_path} (length={len(data)})")
        return data
    except Exception as e:
        logger.error(f"Error reading {relative_path}: {e}")
        raise e

def get_sqlite_connection(relative_path: str) -> sqlite3.Connection:
    """
    Return a connection to an SQLite DB at /data/<relative_path>.
    Logs connection attempts and errors.
    """
    try:
        full_path = safe_join_data(relative_path)
        logger.debug(f"Opening SQLite DB at {full_path}")
        conn = sqlite3.connect(full_path)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite DB {relative_path}: {e}")
        raise e
