from pydantic import StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    collection_metadata: StrictStr = "collection_metadata"

    rage_redis_host: StrictStr = "rage-redis"
    rage_redis_port: int = 6379
    rage_redis_db: int = 0
    rage_redis_username: StrictStr | None = None
    rage_redis_password: StrictStr | None = None

    rage_qdrant_host: StrictStr = "rage-qdrant"
    rage_qdrant_port: int = 6333
    rage_qdrant_grpc_port: int = 6334
    rage_qdrant_api_key: StrictStr | None = None

    dense_embed_doc_cache_path: StrictStr = (
        "/resources/cache/embeddings/documents"
    )
    dense_embed_query_cache_path: StrictStr = (
        "/resources/cache/embeddings/queries"
    )
    fast_embed_sparse_cache: StrictStr = "/resources/cache/fes"

    emb_model: StrictStr = "text-embedding-3-large"
    emb_dimensions: int = 1024

    test_pdf_url: StrictStr = "https://www.argentina.gob.ar/sites/default/files/asi_hablo_zaratustra_nietzsche.pdf"


config = Config()
