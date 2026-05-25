FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y \
    software-properties-common \
    openjdk-8-jdk \
    python3 \
    python3-pip \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
RUN export JAVA_HOME

# Create app directory
RUN mkdir -p /app
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Give permission to start script
RUN chmod +x start.sh

# Default command
CMD ["sh", "start.sh"]