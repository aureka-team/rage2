---
type: Reference
title: Configuration
description: Environment variables used by RAGE.
tags:
  - configuration
  - environment
---

# Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Authentication for OpenAI embeddings. | None |
| `IONOS_TOKEN` | Authentication for `IonosEmbeddings`. | None |
| `RAGE_REDIS_HOST` | Redis host. | `rage-redis` |
| `RAGE_REDIS_PORT` | Redis port. | `6379` |
| `RAGE_REDIS_DB` | Redis database. | `0` |
| `RAGE_QDRANT_HOST` | Qdrant host. | `rage-qdrant` |
| `RAGE_QDRANT_PORT` | Qdrant HTTP port. | `6333` |
| `RAGE_QDRANT_GRPC_PORT` | Qdrant gRPC port. | `6334` |
| `DENSE_EMBED_DOC_CACHE_PATH` | Document embedding cache directory. | `/resources/cache/embeddings/documents` |
| `DENSE_EMBED_QUERY_CACHE_PATH` | Query embedding cache directory. | `/resources/cache/embeddings/queries` |
| `FAST_EMBED_SPARSE_CACHE` | Sparse embedding model cache directory. | `/resources/cache/fes` |
| `EMB_MODEL` | Dense embedding model. | `text-embedding-3-large` |
| `EMB_DIMENSIONS` | Dense embedding dimensions. | `1024` |
| `COLLECTION_METADATA` | Collection used to store collection metadata. | `collection_metadata` |
| `API_PORT` | Published API port. | `8000` |

The settings implementation is in
[`src/rage/config/config.py`](../src/rage/config/config.py).
