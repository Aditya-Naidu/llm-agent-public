# app/tasks.py

import os
import subprocess
import json
import glob
import re
import requests
import time
import math
import csv
from datetime import datetime
from PIL import Image

from .logging_conf import logger
from .config import FALLBACK_EMAIL
from .utils import write_file, read_file, get_sqlite_connection, safe_join_data
from .llm_interface import (
    get_text_embedding,
    analyze_image,
    transcribe_audio,
    using_openai,
    OPENAI_API_KEY,
    AIPROXY_TOKEN,
    CHAT_MODEL,
)

########################
# HELPER TASK FUNCTIONS (Phase A tasks)
########################

def run_a1_install_uv_and_datagen(email: str):
    logger.info("Executing A1: run datagen.py (no uv)")
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    data_path = safe_join_data("")  # e.g. /app/data

    # Download datagen.py
    subprocess.check_call(["wget", "-O", "datagen.py", script_url])

    # Run it with python
    cmd = ["python", "datagen.py", email, "--root", data_path]
    subprocess.check_call(cmd)

    logger.info("A1 completed.")


def run_a2_format_markdown_prettier():
    """
    Format /data/format.md using npx Prettier exactly like the evaluator:
      - Pass file content via stdin
      - Use --stdin-filepath /data/format.md and --parser=markdown
    """
    logger.info("Executing A2: Format /data/format.md with Prettier (parser=markdown)")
    md_file = safe_join_data("format.md")
    if not os.path.exists(md_file):
        logger.warning(f"{md_file} does not exist, skipping A2.")
        return

    original_content = read_file("format.md")
    try:
        process = subprocess.run(
            [
                "npx",
                "prettier@3.4.2",
                "--parser", "markdown",
                "--stdin-filepath", "/data/format.md"
            ],
            input=original_content,
            capture_output=True,
            text=True,
            check=True,
            shell=True,  # use shell if needed
        )
        formatted_output = process.stdout
        if formatted_output.strip():
            write_file("format.md", formatted_output)
            logger.info("A2: Prettier formatting succeeded.")
        else:
            logger.warning("Prettier returned empty output, using fallback naive format.")
            _fallback_markdown_format(md_file)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Prettier formatting failed (A2). stderr:\n{e.stderr}")
        _fallback_markdown_format(md_file)

    logger.info("A2 completed.")


def _fallback_markdown_format(md_file: str):
    logger.warning("Falling back to naive formatting for format.md.")
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = [re.sub(r"\s+", " ", line.strip()) for line in lines]
    with open(md_file, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")


def run_a3_count_wednesdays():
    logger.info("Executing A3: Count Wednesdays")
    in_file = safe_join_data("dates.txt")
    out_file = safe_join_data("dates-wednesdays.txt")
    if not os.path.exists(in_file):
        logger.warning("No dates.txt found.")
        return

    with open(in_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    wednesdays = 0
    possible_formats = ["%Y-%m-%d", "%d-%b-%Y", "%b %d, %Y", "%Y/%m/%d %H:%M:%S"]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed_date = None
        for fmt in possible_formats:
            try:
                parsed_date = datetime.strptime(line, fmt)
                break
            except Exception:
                continue
        if parsed_date and parsed_date.strftime("%A") == "Wednesday":
            wednesdays += 1

    write_file("dates-wednesdays.txt", str(wednesdays))
    logger.info(f"A3 completed. Found {wednesdays} Wednesday(s).")


def run_a4_sort_contacts():
    logger.info("Executing A4: Sort contacts")
    in_file = safe_join_data("contacts.json")
    out_file = safe_join_data("contacts-sorted.json")
    if not os.path.exists(in_file):
        logger.warning("No contacts.json.")
        return

    with open(in_file, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    contacts_sorted = sorted(contacts, key=lambda c: (c["last_name"], c["first_name"]))
    write_file("contacts-sorted.json", json.dumps(contacts_sorted, indent=2))
    logger.info("A4 completed.")


def run_a5_logs_recent():
    logger.info("Executing A5: logs recent")
    logs_dir = safe_join_data("logs")
    out_file = safe_join_data("logs-recent.txt")
    if not os.path.exists(logs_dir):
        logger.warning("No logs directory.")
        return

    all_logs = glob.glob(os.path.join(logs_dir, "*.log"))
    all_logs.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    recent_10 = all_logs[:10]
    lines_out = []
    for lf in recent_10:
        with open(lf, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            lines_out.append(first_line)
    write_file("logs-recent.txt", "\n".join(lines_out))
    logger.info("A5 completed.")


def run_a6_docs_index():
    logger.info("Executing A6: docs index")
    docs_dir = safe_join_data("docs")
    index_file = safe_join_data("docs/index.json")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
    mapping = {}
    for root, dirs, files in os.walk(docs_dir):
        for name in files:
            if name.endswith(".md"):
                md_path = os.path.join(root, name)
                rel_path = os.path.relpath(md_path, docs_dir)
                title = None
                with open(md_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("#"):
                            title = line.lstrip("#").strip()
                            break
                if title:
                    mapping[rel_path] = title
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("A6 completed.")


def run_a7_extract_email_sender():
    logger.info("Executing A7: extract email sender")
    in_file = safe_join_data("email.txt")
    out_file = safe_join_data("email-sender.txt")
    if not os.path.exists(in_file):
        logger.warning("No email.txt found.")
        return
    text = read_file("email.txt")
    match = re.search(r"From:.*?<([^>]+)>", text)
    sender = match.group(1) if match else ""
    write_file("email-sender.txt", sender)
    logger.info("A7 completed.")


def run_a8_extract_credit_card():
    logger.info("Executing A8: extract credit card number from image")
    in_file = safe_join_data("credit_card.png")
    out_file = safe_join_data("credit-card.txt")
    if not os.path.exists(in_file):
        logger.warning("No credit_card.png found.")
        return
    with open(in_file, "rb") as imgf:
        image_bytes = imgf.read()
    prompt = "This is a credit card image. Return JSON with {\"digits\": \"...\"} containing the card number (no spaces)."
    result = _extract_card_with_llm(prompt, image_bytes)
    digits = result.get("digits", "")
    write_file("credit-card.txt", digits)
    logger.info("A8 completed.")


def _extract_card_with_llm(prompt_text, image_bytes):
    import base64
    import json
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "card_extract",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"digits": {"type": "string"}},
                    "required": ["digits"],
                    "additionalProperties": False,
                },
            }
        }
    }
    if using_openai():
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        service_name = "OpenAI-CardExtract"
    else:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
        service_name = "AIProxy-CardExtract"

    max_llm_attempts = 4
    last_digits = ""
    for attempt in range(1, max_llm_attempts + 1):
        logger.info(f"Extracting credit card digits (attempt {attempt}) via {service_name}")
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                rj = resp.json()
                content = rj["choices"][0]["message"]["content"].strip()
                parsed = json.loads(content)
                digits = parsed.get("digits", "").strip()
                logger.info(f"Got digits='{digits}' (length={len(digits)})")
                last_digits = digits
                if len(digits) == 16:
                    return {"digits": digits}
                else:
                    logger.warning(f"digits length != 16 (actual={len(digits)}). Retrying...")
                    time.sleep(1)
                    continue
            else:
                logger.error(f"{service_name} error {resp.status_code}: {resp.text}")
                if attempt == max_llm_attempts:
                    resp.raise_for_status()
                time.sleep(1)
                continue
        except Exception as e:
            logger.error(f"{service_name} attempt {attempt} error: {e}")
            if attempt == max_llm_attempts:
                raise e
            time.sleep(1)
    logger.warning(f"No 16-digit result after {max_llm_attempts} attempts. Using last digits={last_digits}")
    return {"digits": last_digits}


def run_a9_find_similar_comments():
    logger.info("Executing A9: find most similar pair of comments w/ embeddings")
    if using_openai():
        logger.info("A9: Using OpenAI API token for text embeddings.")
    else:
        logger.info("A9: Using AIProxy token for text embeddings.")

    in_file = safe_join_data("comments.txt")
    out_file = safe_join_data("comments-similar.txt")
    if not os.path.exists(in_file):
        logger.warning("No comments.txt found.")
        return
    with open(in_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    if len(comments) < 2:
        write_file("comments-similar.txt", "")
        return

    def cosine_sim(e1, e2):
        dotp = sum(a * b for a, b in zip(e1, e2))
        norm1 = math.sqrt(sum(a * a for a in e1))
        norm2 = math.sqrt(sum(a * a for a in e2))
        return dotp / (norm1 * norm2) if norm1 and norm2 else 0

    comment_embeddings = []
    for c in comments:
        emb = get_text_embedding(c)
        comment_embeddings.append((c, emb))
    best_pair = ("", "")
    best_score = -999
    for i in range(len(comment_embeddings)):
        for j in range(i + 1, len(comment_embeddings)):
            sim = cosine_sim(comment_embeddings[i][1], comment_embeddings[j][1])
            if sim > best_score:
                best_score = sim
                best_pair = (comment_embeddings[i][0], comment_embeddings[j][0])
    write_file("comments-similar.txt", best_pair[0] + "\n" + best_pair[1])
    logger.info(f"A9 completed. Similarity={best_score:.4f}")


def run_a10_ticket_sales_gold():
    logger.info("Executing A10: sum Gold sales in ticket-sales.db")
    db_file = "ticket-sales.db"
    out_file = "ticket-sales-gold.txt"
    conn = None
    try:
        conn = get_sqlite_connection(db_file)
        cur = conn.cursor()
        cur.execute("SELECT SUM(units*price) FROM tickets WHERE type='Gold'")
        result = cur.fetchone()
        total = result[0] if result[0] else 0.0
        write_file(out_file, str(total))
    except Exception as e:
        logger.error(f"A10 DB error: {e}")
    finally:
        if conn:
            conn.close()
    logger.info("A10 completed.")


########################
# Business tasks (Phase B tasks) – Robust implementations
########################

def run_b3_fetch_data_from_api():
    logger.info("Executing B3: fetch data from an API to /data")
    url = "https://jsonplaceholder.typicode.com/posts/1"
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = resp.json()
                    if "id" in data and "title" in data:
                        write_file("api-fetched.json", json.dumps(data, indent=2))
                        logger.info("B3 completed.")
                        return
                    else:
                        logger.warning("B3: JSON structure is unexpected.")
                else:
                    logger.warning("B3: Response is not JSON.")
            else:
                logger.warning(f"B3 API error {resp.status_code}")
        except Exception as e:
            logger.error(f"B3 attempt {attempt} error: {e}")
        time.sleep(1)
    logger.error("B3 failed after multiple attempts.")


def run_b4_clone_git_repo():
    logger.info("Executing B4: clone a git repo")
    repo_url = "https://github.com/octocat/Hello-World.git"
    target_dir = safe_join_data("cloned_repo")
    try:
        if os.path.exists(target_dir):
            try:
                subprocess.run(["git", "pull"], cwd=target_dir, check=True, capture_output=True, text=True)
                logger.info("B4: Repository already exists; pulled latest changes.")
            except Exception as e:
                logger.warning(f"B4: Failed to pull latest changes: {e}. Proceeding without pull.")
        else:
            subprocess.run(["git", "clone", repo_url, target_dir], check=True, capture_output=True, text=True)
            logger.info("B4: Repository cloned successfully.")
        # Configure local git user if not already set
        try:
            subprocess.run(["git", "config", "user.email", "myuser@example.com"], cwd=target_dir, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "My User"], cwd=target_dir, check=True, capture_output=True, text=True)
        except Exception as config_err:
            logger.warning(f"B4: Failed to configure local git user: {config_err}")
        # Create a test commit
        test_file = os.path.join(target_dir, "TEST.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Test commit\n")
        subprocess.run(["git", "add", "TEST.txt"], cwd=target_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "Add test file"], cwd=target_dir, check=True, capture_output=True, text=True)
        logger.info("B4 completed.")
    except subprocess.CalledProcessError as cpe:
        logger.error(f"B4 git error: {cpe.stderr}")
    except Exception as e:
        logger.error(f"B4 error: {e}")


def run_b5_run_sql_query():
    logger.info("Executing B5: run a SQL query on /data/ticket-sales.db")
    db_file = "ticket-sales.db"
    try:
        conn = get_sqlite_connection(db_file)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
            if cur.fetchone() is None:
                logger.error("B5: Table 'tickets' does not exist.")
                return
            cur.execute("SELECT COUNT(*) FROM tickets")
            count_val = cur.fetchone()[0]
        finally:
            conn.close()
        write_file("b5_sql_count.txt", str(count_val))
        logger.info("B5 completed.")
    except Exception as e:
        logger.error(f"B5 DB error: {e}")


def run_b6_scrape_website():
    logger.info("Executing B6: scrape website example")
    url = "https://example.com"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else "No Title"
            if not title:
                title = "No Title"
            write_file("scraped.txt", title)
        else:
            logger.warning(f"B6: Received status code {r.status_code}")
            write_file("scraped.txt", "")
    except Exception as e:
        logger.error(f"B6 scrape error: {e}")
    logger.info("B6 completed.")


def run_b7_compress_or_resize_image():
    logger.info("Executing B7: compress/resize image in /data")
    in_file = safe_join_data("credit_card.png")
    out_file = safe_join_data("credit_card_resized.png")
    if os.path.exists(in_file):
        try:
            with Image.open(in_file) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                new_size = (max(1, img.width // 2), max(1, img.height // 2))
                # Pillow 10 uses Resampling.LANCZOS instead of ANTIALIAS
                if hasattr(Image, "Resampling"):
                    resample_filter = Image.Resampling.LANCZOS
                else:
                    resample_filter = Image.ANTIALIAS
                resized = img.resize(new_size, resample_filter)
                resized.save(out_file)
                logger.info("B7: Image resized and saved successfully.")
        except Exception as e:
            logger.error(f"B7 image processing error: {e}")
    else:
        logger.warning("B7: Input image not found.")
    logger.info("B7 completed.")


def run_b8_transcribe_audio():
    logger.info("Executing B8: transcribe audio from MP3")
    audio_path = safe_join_data("audio.mp3")
    if not os.path.exists(audio_path):
        logger.warning("B8: No audio.mp3 found.")
        return
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            result = transcribe_audio(audio_path)
            if result and "transcription" in result:
                txt = f"Language: {result.get('language', 'unknown')}\n\n{result['transcription']}"
                write_file("audio-transcription.txt", txt)
                logger.info("B8 completed.")
                return
        except Exception as e:
            logger.error(f"B8 audio transcription error on attempt {attempt}: {e}")
        time.sleep(2)
    logger.error("B8: Failed to transcribe audio after multiple attempts.")


def run_b9_convert_md_to_html():
    logger.info("Executing B9: convert Markdown to HTML")
    md_file = safe_join_data("format.md")
    out_file = safe_join_data("format.html")
    if not os.path.exists(md_file):
        write_file("format.html", "<p>No format.md found</p>")
        return
    try:
        content = read_file("format.md")
        try:
            import markdown
            html = markdown.markdown(content)
        except ImportError:
            logger.warning("B9: 'markdown' library not found, using fallback conversion.")
            lines = content.splitlines()
            html_lines = []
            for line in lines:
                if line.startswith("#"):
                    line = "<h1>" + line.lstrip("#").strip() + "</h1>"
                html_lines.append(line)
            html = "\n".join(html_lines)
        write_file("format.html", html)
        logger.info("B9 completed.")
    except Exception as e:
        logger.error(f"B9 conversion error: {e}")


def run_b10_api_endpoint_filter_csv():
    logger.info("Executing B10: filter a CSV in /data/file.csv => /data/file-filtered.json")
    csv_file = safe_join_data("file.csv")
    out_file = safe_join_data("file-filtered.json")
    if not os.path.exists(csv_file):
        logger.warning("B10: No file.csv found.")
        return
    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logger.error("B10: CSV file has no headers.")
                return
            first_field = reader.fieldnames[0]
            filtered = [row for row in reader if row.get(first_field, "").strip() == "foo"]
        write_file("file-filtered.json", json.dumps(filtered, indent=2))
        logger.info("B10 completed.")
    except Exception as e:
        logger.error(f"B10 error: {e}")


########################
# MAIN DISPATCH
########################

def execute_tasks(task_ids, email: str):
    if not email:
        email = FALLBACK_EMAIL
    for t in task_ids:
        if t == "A1": run_a1_install_uv_and_datagen(email)
        elif t == "A2": run_a2_format_markdown_prettier()
        elif t == "A3": run_a3_count_wednesdays()
        elif t == "A4": run_a4_sort_contacts()
        elif t == "A5": run_a5_logs_recent()
        elif t == "A6": run_a6_docs_index()
        elif t == "A7": run_a7_extract_email_sender()
        elif t == "A8": run_a8_extract_credit_card()
        elif t == "A9": run_a9_find_similar_comments()
        elif t == "A10": run_a10_ticket_sales_gold()
        elif t == "B3": run_b3_fetch_data_from_api()
        elif t == "B4": run_b4_clone_git_repo()
        elif t == "B5": run_b5_run_sql_query()
        elif t == "B6": run_b6_scrape_website()
        elif t == "B7": run_b7_compress_or_resize_image()
        elif t == "B8": run_b8_transcribe_audio()
        elif t == "B9": run_b9_convert_md_to_html()
        elif t == "B10": run_b10_api_endpoint_filter_csv()
        else:
            logger.info(f"Unrecognized task ID: {t}")
