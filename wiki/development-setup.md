---
type: Guide
title: Development setup
description: Run RAGE and its supporting services locally.
tags:
  - development
  - docker
  - setup
---

# Development setup

The shared [`Dockerfile`](../Dockerfile) defines `core`, `api`,
and `devcontainer` build targets so local development and the deployed API use
the same operating system, Python runtime, system packages, and Python
dependencies.

Run the project inside the devcontainer.

Start Qdrant:

```bash
make qdrant-start
```

Start Redis when using loader caching:

```bash
make redis-start
```

Build and run the API:

```bash
make api-run
```

Alternatively, run it in the background:

```bash
make api-up
```
