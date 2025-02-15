# app/logging_conf.py

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,  # set to DEBUG to log everything possible
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("automation-agent")
