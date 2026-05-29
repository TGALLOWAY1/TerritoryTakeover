# Territory Takeover Arena — container image for cloud deploys.
#
# Build:  docker build -t tt-arena .
# Run:    docker run -e TT_ARENA_TOKEN=secret -p 8000:8000 tt-arena
# Then open http://localhost:8000/?token=secret
#
# The arena is a single stateful process (game state + a background game
# thread live in memory), so run exactly ONE instance — never scale to
# multiple replicas, which would split state across processes.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

# Bind to all interfaces inside the container; the platform provides $PORT
# and (optionally) $TT_ARENA_TOKEN at runtime.
ENV TT_HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "scripts/play_interactive.py", "--no-browser"]
