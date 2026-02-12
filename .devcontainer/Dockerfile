FROM mcr.microsoft.com/devcontainers/python:0-3.11

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Hard fix: remove Yarn repo even if injected by devcontainer features
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && rm -f /etc/apt/trusted.gpg.d/yarn.gpg \
    && rm -f /usr/share/keyrings/yarn.gpg || true

# Update & install safely
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl gnupg \
       postgresql-client \
    && rm -rf /var/lib/apt/lists/*