# Container for the RAG Eval Judge API.
# Ollama runs as a SEPARATE container (see docker-compose.yml) — this image
# only holds the Python service and talks to Ollama over the network.
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY rag_eval/ ./rag_eval/
COPY evaluation/ ./evaluation/
COPY app/ ./app/

# Point the Ollama client at the ollama service; cache HF models inside image fs.
ENV OLLAMA_HOST=http://ollama:11434
ENV HF_HOME=/app/.hf-cache

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
