FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3==1.34.0 \
    psycopg2-binary==2.9.9 \
    pymysql==1.1.0

# Create a non-root user
RUN useradd -m -u 1000 mlflow
USER mlflow

# Set working directory
WORKDIR /home/mlflow

# Expose port
EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]