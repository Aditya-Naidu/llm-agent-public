# LLM-based Automation Agent

This repository contains a FastAPI application that can interpret plain-English tasks via GPT-4o-Mini and automate them. It is structured to store all data under `./data`, never delete data, and never access data outside `./data`.
This FastAPI project parses user tasks, identifies them as A1..A10 or B3..B10, and executes them. It uses structured LLM calls (chat, embeddings, image, audio) with fallback logic: if `OPENAI_API_KEY` is present, it uses official OpenAI; else it uses `AIPROXY_TOKEN`.

## Requirements

- Python 3.10+
- Docker (if you want to build and run in a container)
- An AI Proxy Token (AIPROXY_TOKEN) for GPT-4o-Mini

## Setup

1. **Clone** this repo.
2. **Create a `.env`** file in the root:
   ```bash
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
   AIPROXY_TOKEN="YOUR_AIPROXY_TOKEN_HERE"
