from pydantic import StrictInt, StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_host: StrictStr = "rage-redis"
    redis_port: StrictInt = 6379
    redis_db: StrictInt = 0

    qdrant_host: StrictStr = "rage-qdrant"
    qdrant_port: StrictInt = 6333
    qdrant_grpc_port: StrictInt = 6334

    dense_embed_doc_cache_path: StrictStr = (
        "/resources/cache/embeddings/documents"
    )

    dense_embed_query_cache_path: StrictStr = (
        "/resources/cache/embeddings/queries"
    )

    fast_embed_sparse_cache: StrictStr = "/resources/cache/fes"


config = Config()
