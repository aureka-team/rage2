from pydantic import PositiveInt, StrictInt, StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    collection_metadata: StrictStr = "collection_metadata"

    rage_redis_host: StrictStr = "rage-redis"
    rage_redis_port: StrictInt = 6379
    rage_redis_db: StrictInt = 0

    rage_qdrant_host: StrictStr = "rage-qdrant"
    rage_qdrant_port: StrictInt = 6333
    rage_qdrant_grpc_port: StrictInt = 6334

    dense_embed_doc_cache_path: StrictStr = (
        "/resources/cache/embeddings/documents"
    )
    dense_embed_query_cache_path: StrictStr = (
        "/resources/cache/embeddings/queries"
    )
    fast_embed_sparse_cache: StrictStr = "/resources/cache/fes"

    emb_model: StrictStr = "text-embedding-3-large"
    emb_dimensions: PositiveInt = 1024

    test_pdf_url: StrictStr = "https://www.argentina.gob.ar/sites/default/files/asi_hablo_zaratustra_nietzsche.pdf"


config = Config()
