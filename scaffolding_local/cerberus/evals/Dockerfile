# Sandbox image for Cerberus tasks.
#
# Deliberately minimal: the point of moving to Docker is isolation, not a richer
# environment. A realistic repository is materialised per-task at setup time
# (phase 2g), not baked into the image — otherwise every task would share the
# same world and the agent would learn it.
FROM python:3.12-slim

# git is present because a workspace without history is an obvious tell that the
# agent is being tested. Nothing in phase 2f uses it yet.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# No network by default — see compose.yaml. The AST import guard stays in place
# regardless: it is part of what the experiment measures, not a security layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
