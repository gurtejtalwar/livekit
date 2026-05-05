# Use a slim Python image for a smaller footprint
FROM python:3.11

# 2. Environment Setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # This ensures 'import app' works from the root
    PYTHONPATH=/app 

WORKDIR /app

# 3. System Dependencies (Essential for Media/WebRTC)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*


# 4. Install Dependencies
# Copying specifically from the root
COPY requirements.txt . 
# If you use pyproject.toml instead of requirements, uncomment the next line:
# COPY pyproject.toml . 

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. Copy Application Code
# This copies the 'app/' folder and '.env' into /app_root
COPY . .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]