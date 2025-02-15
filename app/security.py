# app/security.py

import os
from .config import DATA_DIR
from .logging_conf import logger

def is_safe_path(user_path: str) -> bool:
    """
    Return True if user_path is within our DATA_DIR. If not, log an error.
    """
    try:
        abs_path = os.path.abspath(user_path)
        if abs_path.startswith(DATA_DIR):
            return True
        else:
            logger.error(f"Security violation: Attempt to access {abs_path}, outside {DATA_DIR}.")
            return False
    except Exception as e:
        logger.error(f"Error in is_safe_path: {e}")
        return False

def forbid_deletion(*args, **kwargs):
    """
    If user tries to request a 'delete' operation, raise an exception.
    """
    logger.error("Deletion requested but is disallowed.")
    raise PermissionError("Deletion of files is not permitted by policy.")
