# app/main.py

from fastapi import FastAPI, Query, Response
from fastapi.responses import PlainTextResponse
import traceback

from .logging_conf import logger
from .llm_interface import parse_user_task_with_llm
from .tasks import execute_tasks
from .utils import read_file
from .config import FALLBACK_EMAIL

app = FastAPI()

@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    """
    POST /run?task=...
    - parse with parse_user_task_with_llm
    - call execute_tasks
    - return 200, 400, or 500
    """
    try:
        logger.info(f"Received /run request with task={task}")
        parsed = parse_user_task_with_llm(task)
        task_ids = parsed.get("task_ids", [])
        email = parsed.get("email", FALLBACK_EMAIL)

        if not task_ids:
            return Response("No recognized tasks from your request", status_code=400)

        execute_tasks(task_ids, email)
        return Response("Tasks executed successfully", status_code=200)

    except ValueError as ve:
        logger.error(f"User error: {ve}")
        return Response(f"User Error: {ve}", status_code=400)
    except PermissionError as pe:
        logger.error(f"Security error: {pe}")
        return Response(f"Security Error: {pe}", status_code=400)
    except Exception as e:
        logger.error(f"Agent error: {traceback.format_exc()}")
        return Response("Internal Server Error", status_code=500)

@app.get("/read")
async def read_file_contents(path: str = Query(..., description="Path under /data")):
    """
    GET /read?path=...
    Returns 200+file content if found, else 404.
    The evaluator calls e.g. /read?path=/data/format.md
    We'll strip off any leading "/data/" before passing to our read logic.
    """
    try:
        logger.info(f"Received /read request for path={path}")

        # If the user gave something like "/data/format.md", we remove the leading "/data/"
        if path.startswith("/data/"):
            path = path[len("/data/") :]

        # Also, if there's a leading slash after that, remove it
        if path.startswith("/"):
            path = path[1:]

        content = read_file(path)
        return PlainTextResponse(content, status_code=200)

    except FileNotFoundError:
        logger.warning("File not found.")
        return Response(status_code=404)
    except PermissionError as pe:
        logger.error(f"Security error: {pe}")
        return Response(str(pe), status_code=400)
    except Exception:
        logger.error(f"Agent error: {traceback.format_exc()}")
        return Response("Internal Server Error", status_code=500)
