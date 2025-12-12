FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
# Setup dòng dưới để cài torch với CPU Render
# RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Setup 2 dòng dưới để tải model từ Google Drive 
# RUN pip install gdown
# RUN gdown --id "1SlZVGO_NuN2022TV3j28uq_jwfbev6yk" -O model.pt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
# Setup dòng dưới này khi deploy lên Render
# CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"] 