# Use conda-forge base image with CUDA support
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment file
COPY environment.yml .

# Create conda environment from your exact specification
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "classifyenv", "/bin/bash", "-c"]

# Verify installation
RUN conda run -n classifyenv python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
RUN mkdir -p data model src

# Copy source code
COPY src/ ./src/
COPY run.py .
COPY run.sh .

# Make run.sh executable
RUN chmod +x run.sh

# Create data and model directories with proper permissions
RUN mkdir -p /app/data /app/model && \
    chmod 755 /app/data /app/model

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose any ports if needed (optional)
# EXPOSE 8000

# Default command - activate conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "classifyenv", "bash"]