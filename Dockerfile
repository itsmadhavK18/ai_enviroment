FROM python:3.10-slim

WORKDIR /app

# Install standard dependencies according to OpenEnv specification
RUN pip install --no-cache-dir pydantic openai openenv-core

# Copy the environment source
COPY . /app

# Run inference script by default
CMD ["python", "inference.py"]
