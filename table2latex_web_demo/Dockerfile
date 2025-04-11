FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    texlive-full \
    poppler-utils \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8396

CMD ["python", "app.py"]
