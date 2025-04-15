FROM rage-core

COPY rage /src/rage
COPY requirements.txt pyproject.toml  /src/
RUN --mount=type=bind,source=.git,target=/src/.git \
    uv pip install /src --no-deps --system

WORKDIR /src/rage/api

ARG API_PORT
CMD fastapi run app.py --port ${API_PORT}
