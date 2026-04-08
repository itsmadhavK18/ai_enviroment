FROM python:3.10-slim

WORKDIR /app

# Copy the environment source
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run Gradio app for Hugging Face Space
CMD ["python", "app.py"]
