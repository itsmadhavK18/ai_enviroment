FROM python:3.10-slim

WORKDIR /app

# Copy the environment source
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run FastAPI + Gradio app for Hugging Face Space
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
