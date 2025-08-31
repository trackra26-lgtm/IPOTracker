# Base image includes Python + Chromium + Playwright browsers preinstalled
FROM mcr.microsoft.com/playwright/python:v1.43.0-noble

# Make Python output unbuffered & no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Playwright browsers already installed in this image, so no playwright install needed

# Render (and many PaaS) provide $PORT
ENV PORT=8000

# Expose for local testing (Render ignores EXPOSE, but it's handy locally)
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
