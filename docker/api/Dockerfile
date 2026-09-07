FROM rage-core

COPY src /src/src
COPY README.md requirements.txt pyproject.toml /src/
RUN --mount=type=bind,target=/root/.cache,source=.git,target=/src/.git \
    uv pip install /src --no-deps

WORKDIR /src/src/rage/api

ARG API_PORT
CMD fastapi run app.py --port ${API_PORT}
