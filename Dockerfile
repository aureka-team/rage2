ARG UV_VERSION=0.12.8
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_source

FROM ubuntu:resolute AS core

ARG PYTHON_VERSION=3.14
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    curl \
    git-core \
    openssh-client \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-setuptools \
    ca-certificates \
    pandoc \
    libreoffice-writer \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

RUN rm /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1
COPY --from=uv_source /uv /uvx /bin/

WORKDIR /tmp

COPY uv.toml .
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    uv pip install -r requirements.txt

RUN rm -rf /tmp/*
WORKDIR /root

FROM core AS api

COPY src /src/src
COPY README.md requirements.txt pyproject.toml /src/
RUN --mount=type=bind,source=.git,target=/src/.git \
    --mount=type=cache,target=/root/.cache \
    uv pip install /src --no-deps

WORKDIR /src/src/rage/api

CMD ["fastapi", "run", "app.py", "--port", "8000"]

FROM core AS devcontainer

WORKDIR /tmp

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    zsh \
    sudo \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

ARG DEVCONTAINER_USER=ubuntu
RUN usermod -aG sudo $DEVCONTAINER_USER \
    && passwd -d $DEVCONTAINER_USER

ENV SHELL=/usr/bin/zsh
RUN chsh $DEVCONTAINER_USER -s $SHELL

WORKDIR /workspace

USER $DEVCONTAINER_USER
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
