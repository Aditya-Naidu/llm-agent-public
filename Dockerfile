# Dockerfile

# 1) Start from a minimal Python image
FROM python:3.10-slim

# 2) Install OS packages: nodejs, npm, git, wget, etc.
#    Then install Prettier globally.
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    git \
    wget \
 && npm install -g prettier@3.4.2 \
 && rm -rf /var/lib/apt/lists/*

# 3) Set a working directory for your app code
WORKDIR /app

# 4) Copy in your requirements first so Docker can cache the pip install step
COPY requirements.txt /app/requirements.txt

# 5) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your app code
COPY . /app

# 7) Expose port 8000 for FastAPI
EXPOSE 8000

# 8) Set default environment variables for tokens (user overrides with -e)
ENV AIPROXY_TOKEN=""
ENV OPENAI_API_KEY=""

# 9) Launch the app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
