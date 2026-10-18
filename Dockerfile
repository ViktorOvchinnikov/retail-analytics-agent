FROM python:3.11-slim

ARG APP_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY retail_agent ./retail_agent
COPY evals ./evals

RUN pip install . \
    && useradd --create-home --uid "${APP_UID}" --shell /usr/sbin/nologin retailagent \
    && mkdir -p /var/lib/retail-agent /home/retailagent/.config/gcloud \
    && chown -R retailagent:retailagent /var/lib/retail-agent /home/retailagent

USER retailagent
WORKDIR /var/lib/retail-agent

ENTRYPOINT ["retail-agent"]
CMD ["--user", "manager_1", "--conversation", "docker-demo"]
