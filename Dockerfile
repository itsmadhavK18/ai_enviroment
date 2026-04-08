FROM python:3.10-slim

WORKDIR /app

# Copy the environment source
COPY . /app

# Install dependencies using the new pyproject.toml
RUN pip install --no-cache-dir .

# Run FastAPI + Gradio app for Hugging Face Space
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
