# app/llm_interface.py

import os
import json
import time
import base64
import requests
from .config import OPENAI_API_KEY, AIPROXY_TOKEN
from .logging_conf import logger

#############################
# Load system_prompt from system_prompt.txt
#############################
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
if not os.path.exists(PROMPT_FILE_PATH):
    raise FileNotFoundError(f"System prompt file not found: {PROMPT_FILE_PATH}")

with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

#############################
# Constants
#############################
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
OPENAI_AUDIO_URL = "https://api.openai.com/v1/audio/transcriptions"

AIPROXY_CHAT_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_EMBED_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

# We'll default to "gpt-4o-mini" if using AI Proxy
# If you have GPT-4 or gpt-3.5-turbo in official OpenAI, you can swap that in if you prefer
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
WHISPER_MODEL = "whisper-1"

#############################
# Helper: which service do we use?
#############################
def using_openai() -> bool:
    """Return True if OPENAI_API_KEY is non-empty (we prefer it)."""
    return bool(OPENAI_API_KEY.strip())

#############################
# 1) parse_user_task_with_llm
#############################
def parse_user_task_with_llm(task_description: str) -> dict:
    """
    Classify the user's request into known tasks A1..A10, B3..B10 + email.
    Always returns JSON with {"task_ids": [...], "email": "..."}.

    Uses the system_prompt from system_prompt.txt, plus the "response_format" block
    to ensure we get structured JSON from the model.

    2 attempts, fallback to AI Proxy if OPENAI_API_KEY isn't present.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_description}
    ]

    data = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "task_parse",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "email": {"type": "string"}
                    },
                    "required": ["task_ids", "email"],
                    "additionalProperties": False
                }
            }
        }
    }

    if using_openai():
        url = OPENAI_CHAT_URL
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        service_name = "OpenAI"
    else:
        url = AIPROXY_CHAT_URL
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        service_name = "AIProxy"

    for attempt in range(1, 3):
        try:
            logger.info(f"Parsing user task (attempt {attempt}) via {service_name} chat.")
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                rj = resp.json()
                content = rj["choices"][0]["message"]["content"].strip()
                try:
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    logger.error(f"Model returned invalid JSON: {content}")
                    raise ValueError("LLM did not return valid JSON.")
            else:
                logger.error(f"{service_name} chat error {resp.status_code}: {resp.text}")
                if attempt == 2:
                    resp.raise_for_status()
        except Exception as e:
            logger.error(f"Error calling {service_name} on attempt {attempt}: {e}")
            if attempt == 2:
                raise e
        time.sleep(1)

    raise RuntimeError("parse_user_task_with_llm failed after 2 attempts.")

#############################
# 2) analyze_image
#############################
def analyze_image(
    prompt_text: str,
    image_bytes: bytes,
    image_format: str = "png"
) -> dict:
    """
    Analyze an image (provided as raw bytes). Returns a structured JSON object
    describing what is in the image. We send a chat request with "response_format".

    - If we have OPENAI_API_KEY, we use official OpenAI. Otherwise AIProxy
    - You can define your own schema in 'response_format' to ensure structured output.

    Returns a dict with your chosen keys, e.g.:
    {
        "description": "some text describing the image",
        "objects": ["cat", "dog", ...],
        ...
    }
    """
    # Convert to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # We embed the question + a structured request.
    # We'll ask "What is in this image?" or any prompt_text provided,
    # plus we pass the image as a "image_url" with data: URI
    # Then we request a JSON schema for consistent output

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    # Example: we want a structured output describing "summary" and "objects".
    data = {
        "model": CHAT_MODEL,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "objects": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["summary", "objects"],
                    "additionalProperties": False
                }
            }
        }
    }

    if using_openai():
        url = OPENAI_CHAT_URL
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        service_name = "OpenAI ImageChat"
    else:
        url = AIPROXY_CHAT_URL
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        service_name = "AIProxy ImageChat"

    for attempt in range(1, 3):
        try:
            logger.info(f"Analyzing image (attempt {attempt}) via {service_name}.")
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                rj = resp.json()
                content = rj["choices"][0]["message"]["content"].strip()
                try:
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    logger.error(f"{service_name} returned invalid JSON: {content}")
                    raise ValueError("Image LLM: invalid JSON response.")
            else:
                logger.error(f"{service_name} image analysis error {resp.status_code}: {resp.text}")
                if attempt == 2:
                    resp.raise_for_status()
        except Exception as e:
            logger.error(f"Error calling {service_name} (image) on attempt {attempt}: {e}")
            if attempt == 2:
                raise e
        time.sleep(1)

    raise RuntimeError("analyze_image failed after 2 attempts.")

#############################
# 3) transcribe_audio
#############################
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file with OpenAI's Whisper.
    Only works if OPENAI_API_KEY is present (AI Proxy doesn't do audio).

    We also do structured output if the API supports "response_format = json".
    The official docs:
      POST /v1/audio/transcriptions
      "response_format" can be "json", "verbose_json", "srt", "vtt", or "text".

    We'll request JSON for consistency:
    => returns { "text": "..." }

    We'll wrap that in our own dict if needed, e.g. { "transcription": "...", "language": "..." }.

    *Raise an error* if OPENAI_API_KEY is empty.
    """
    if not using_openai():
        raise RuntimeError("Audio transcription is only supported with an OpenAI API key.")

    url = OPENAI_AUDIO_URL
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": WHISPER_MODEL,
        "response_format": "json"
    }

    # Attempt up to 2 times
    for attempt in range(1, 3):
        try:
            logger.info(f"Transcribing audio (attempt {attempt}) with Whisper.")
            with open(audio_path, "rb") as audio_file:
                files = {
                    "file": (os.path.basename(audio_path), audio_file, "audio/mpeg")
                }
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            if resp.status_code == 200:
                rj = resp.json()
                # Officially, "text" is the main field if "response_format= json".
                # We'll wrap it in a structured dict if you want more fields:
                return {
                    "transcription": rj.get("text", ""),
                    "language": rj.get("language", "unknown")
                }
            else:
                logger.error(f"OpenAI audio error {resp.status_code}: {resp.text}")
                if attempt == 2:
                    resp.raise_for_status()
        except Exception as e:
            logger.error(f"Audio transcription error attempt {attempt}: {e}")
            if attempt == 2:
                raise e
        time.sleep(2)

    raise RuntimeError("transcribe_audio failed after 2 attempts.")

#############################
# 4) get_text_embedding
#############################
def get_text_embedding(text: str) -> list:
    """
    Return a vector embedding for 'text'.
    If OPENAI_API_KEY is present, use official OpenAI embeddings,
    else fallback to AI Proxy.
    2 attempts, returns the embedding vector as a list of floats.
    """
    data = {
        "model": EMBED_MODEL,
        "input": text
    }

    if using_openai():
        url = OPENAI_EMBED_URL
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        service_name = "OpenAI Embeddings"
    else:
        url = AIPROXY_EMBED_URL
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        service_name = "AIProxy Embeddings"

    for attempt in range(1, 3):
        try:
            logger.info(f"Getting embeddings (attempt {attempt}) from {service_name}.")
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            if resp.status_code == 200:
                rj = resp.json()
                emb = rj["data"][0]["embedding"]
                return emb
            else:
                logger.error(f"{service_name} error {resp.status_code}: {resp.text}")
                if attempt == 2:
                    resp.raise_for_status()
        except Exception as e:
            logger.error(f"Embedding request error attempt {attempt}: {e}")
            if attempt == 2:
                raise e
        time.sleep(1)

    raise RuntimeError("get_text_embedding failed after 2 attempts.")
