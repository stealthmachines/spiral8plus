# Grand Unified Theory - High-Precision Testing Environment
# ==========================================================
# Multi-stage Docker build for maximum performance and minimal size

# Stage 1: Build C precision engine
FROM gcc:12 as builder

WORKDIR /build

# Copy C source
COPY gut_precision_engine.c .
COPY hdgl_analog_v30b.c .

# Compile with maximum optimization
RUN gcc -O3 -march=native -funroll-loops -ffast-math \
    -o gut_engine gut_precision_engine.c -lm && \
    gcc -O3 -march=native -funroll-loops \
    -shared -fPIC -o libhdgl.so hdgl_analog_v30b.c -lm

# Stage 2: Python scientific environment
FROM python:3.11-slim

LABEL maintainer="GUT Research Team"
LABEL description="Grand Unified Theory Testing Environment"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /gut

# Copy compiled C binaries from builder
COPY --from=builder /build/gut_engine /usr/local/bin/
COPY --from=builder /build/libhdgl.so /usr/local/lib/
RUN ldconfig

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python framework (required files)
COPY grand_unified_theory.py gut_data_analysis.py gut_demo.py ./
COPY data_inventory.py download_data.py ./
COPY validate_trusted_data.py tune_echo_parameters.py test_tuned_ligo.py ./
COPY tuned_echo_parameters.json ./

# Copy documentation
COPY physics.md TUNING_RESULTS.md ./

# Create data directories (mount externally if needed)
RUN mkdir -p /gut/bigG /gut/micro-bot-digest /gut/hdgl_harmonics_spiral10000_analog_v30

# Create output directories
RUN mkdir -p /gut/output /gut/plots /gut/logs /gut/ligo_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV GUT_DATA_DIR=/gut
ENV GUT_OUTPUT_DIR=/gut/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import numpy, scipy, matplotlib; print('OK')" || exit 1

# Default command: run comprehensive validation
CMD ["python", "grand_unified_theory.py"]

# Alternative entry points (use docker run gut-testing <command>):
# - demo: python gut_demo.py
# - analyze: python gut_data_analysis.py
# - c-engine: gut_engine validate-all
# - download: python download_data.py
# - shell: /bin/bash
