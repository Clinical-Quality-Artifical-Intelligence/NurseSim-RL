# Use Python 3.11 base image (required for agentbeats SDK)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make run.sh executable
RUN chmod +x run.sh

# Expose ports
# 7860: Gradio demo
# 8080: AgentBeats A2A controller
EXPOSE 7860 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_MODE=a2a

# Run the application via launcher script
CMD ["./run.sh"]
