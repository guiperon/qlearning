FROM python:3.12-slim-bookworm

WORKDIR /qlearning/app

COPY requirements.txt .

# update OS packages and upgrade pip tooling to reduce known vulnerabilities
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        vim \
        procps \
        iputils-ping && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

COPY app /qlearning/app

CMD ["python", "/app/ThroughputPower.py"]