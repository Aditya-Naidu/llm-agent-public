# app/chunking.py

from .logging_conf import logger

def chunk_text(text: str, max_size: int = 2048):
    """
    Yields consecutive chunks of `text` of up to `max_size` characters.
    Logs debugging info if text is large or if chunking occurs.
    """
    if len(text) <= max_size:
        logger.debug("No need to chunk; text is within max_size.")
        yield text
        return

    logger.debug(f"Chunking text of length {len(text)} into max {max_size} sized pieces.")
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        chunk = text[start:end]
        yield chunk
        logger.debug(f"Emitted chunk from {start} to {end} (length {len(chunk)}).")
        start += max_size
