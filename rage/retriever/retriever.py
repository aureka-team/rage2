import os

from uuid import uuid4
from functools import lru_cache
from common.logger import get_logger

from pydantic import (
    BaseModel,
    StrictStr,
    NonNegativeFloat,
    StrictFloat,
    StrictInt,
    Field,
)

from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.conversions.common_types import PointId

from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_classic.embeddings import CacheBackedEmbeddings

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from rage.meta.interfaces import TextChunk


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

DENSE_EMBED_DOC_CACHE_PATH = os.getenv("DENSE_EMBED_DOC_CACHE_PATH")
DENSE_EMBED_QUERY_CACHE_PATH = os.getenv("DENSE_EMBED_QUERY_CACHE_PATH")
FAST_EMBED_SPARSE_CACHE = os.getenv("FAST_EMBED_SPARSE_CACHE")


logger = get_logger(__name__)


class RetrieverItem(BaseModel):
    text: StrictStr
    metadata: dict
    score: NonNegativeFloat | None = None


class WeightedMetadataItem(BaseModel):
    key: StrictStr
    value: StrictStr | StrictInt | StrictFloat
    weight: NonNegativeFloat = Field(le=1.0)


class Retriever:
    def __init__(
        self,
        dense_embeddings: Embeddings,
        dense_embed_doc_cache_path: str | None = DENSE_EMBED_DOC_CACHE_PATH,
        dense_embed_query_cache_path: str | None = DENSE_EMBED_QUERY_CACHE_PATH,
        sparse_embed_model_name: str = "Qdrant/bm25",
    ):
        assert dense_embeddings.dimensions is not None, (  # type: ignore
            "Expected 'dense_embeddings.dimensions' to be set."
        )

        assert dense_embeddings.model is not None, (  # type: ignore
            "Expected 'dense_embeddings.model' to be set."
        )

        self.dense_embed_dimensions = dense_embeddings.dimensions
        self.dense_embeddings = self._get_dense_embeddings(
            dense_embeddings=dense_embeddings,
            dense_embed_doc_cache_path=dense_embed_doc_cache_path,
            dense_embed_query_cache_path=dense_embed_query_cache_path,
        )

        self.sparse_embeddings = FastEmbedSparse(
            model_name=sparse_embed_model_name,
            cache_dir=FAST_EMBED_SPARSE_CACHE,
        )

        self.qadrant_client = QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
        )

        self.qadrant_async_client = AsyncQdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
        )

    def _get_dense_embeddings(
        self,
        dense_embeddings: Embeddings,
        dense_embed_doc_cache_path: str | None,
        dense_embed_query_cache_path: str | None,
    ) -> Embeddings:
        if dense_embed_doc_cache_path is None:
            return dense_embeddings

        query_embedding_cache = (
            True
            if dense_embed_query_cache_path is None
            else LocalFileStore(root_path=dense_embed_query_cache_path)
        )

        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=dense_embeddings,
            document_embedding_cache=LocalFileStore(
                root_path=dense_embed_doc_cache_path
            ),
            namespace=dense_embeddings.model,  # type: ignore
            query_embedding_cache=query_embedding_cache,
        )

    @lru_cache()
    def _get_dense_vector_store(
        self,
        collection_name: str,
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qadrant_client,
            collection_name=collection_name,
            embedding=self.dense_embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )

    @lru_cache()
    def _get_hybrid_vector_store(
        self,
        collection_name: str,
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qadrant_client,
            collection_name=collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    async def create_collection(self, collection_name: str) -> None:
        if await self.qadrant_async_client.collection_exists(
            collection_name=collection_name
        ):
            logger.warning(f"collection {collection_name} already exists.")
            return

        await self.qadrant_async_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_embed_dimensions,  # type: ignore
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    async def insert_text_chunks(
        self,
        collection_name: str,
        text_chunks: list[TextChunk],
        batch_size: int = 256,
    ) -> None:
        if not self.qadrant_client.collection_exists(
            collection_name=collection_name
        ):
            logger.warning(f"collection {collection_name} doesn't exists.")
            return

        vector_store = self._get_hybrid_vector_store(
            collection_name=collection_name
        )

        lg_documents = [
            Document(
                page_content=tc.text,
                metadata=tc.metadata,
            )
            for tc in text_chunks
        ]

        uuids = [str(uuid4()) for _ in range(len(lg_documents))]
        await vector_store.aadd_documents(
            documents=lg_documents,
            ids=uuids,
            batch_size=batch_size,
        )

    def _parse_results(
        self,
        results: list[tuple[Document, float]],
    ) -> list[RetrieverItem]:
        return [
            RetrieverItem(
                text=document.page_content,
                metadata=document.metadata,
                score=score,
            )
            for document, score in results
        ]

    async def dense_search(
        self,
        collection_name: str,
        query: str,
        k: int = 10,
        score_threshold: float | None = None,
        search_filter: models.Filter | None = None,
    ) -> list[RetrieverItem]:
        vector_store = self._get_dense_vector_store(
            collection_name=collection_name
        )

        results = await vector_store.asimilarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter=search_filter,
        )

        return self._parse_results(results=results)

    async def dense_search_batch(
        self,
        collection_name: str,
        queries: list[str],
        k: int = 10,
        score_threshold: float | None = None,
        search_filter: models.Filter | None = None,
    ) -> list[list[RetrieverItem]]:
        vectors = await self.dense_embeddings.aembed_documents(texts=queries)
        requests = [
            models.QueryRequest(
                query=vector,
                using="dense",
                limit=k,
                filter=search_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            for vector in vectors
        ]

        query_responses = await self.qadrant_async_client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )

        retriever_items = [
            [
                RetrieverItem(
                    text=points.payload["page_content"],  # type: ignore
                    metadata=points.payload["metadata"],  # type: ignore
                    score=points.score,
                )
                for points in qr.points
            ]
            for qr in query_responses
        ]

        return retriever_items

    async def hybrid_search(
        self,
        collection_name: str,
        query: str,
        k: int = 10,
        score_threshold: float | None = None,
        search_filter: models.Filter | None = None,
    ) -> list[RetrieverItem]:
        vector_store = self._get_hybrid_vector_store(
            collection_name=collection_name
        )

        results = await vector_store.asimilarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter=search_filter,
        )

        return self._parse_results(results=results)

    async def scroll(
        self,
        collection_name: str,
        limit: int = 10,
        scroll_filter: models.Filter | None = None,
        order_by: models.OrderBy | None = None,
        offset: PointId | None = None,
    ) -> list[models.Record]:
        results = await self.qadrant_async_client.scroll(
            collection_name=collection_name,
            limit=limit,
            scroll_filter=scroll_filter,
            order_by=order_by,
            offset=offset,
        )

        if results is None:
            return []

        return results[0]

    async def delete_chunks(
        self,
        collection_name: str,
        key: str,
        value: str | int | bool,
    ) -> None:
        if not await self.qadrant_async_client.collection_exists(
            collection_name=collection_name
        ):
            logger.warning(f"collection {collection_name} doesn't exist.")
            return

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            ]
        )

        await self.qadrant_async_client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=delete_filter),
        )

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: models.PayloadSchemaType = models.PayloadSchemaType.KEYWORD,
    ) -> None:
        if not await self.qadrant_async_client.collection_exists(
            collection_name=collection_name
        ):
            logger.warning(f"collection {collection_name} doesn't exist.")
            return

        collection_info = await self.qadrant_async_client.get_collection(
            collection_name=collection_name
        )

        existing_indexes = set(collection_info.payload_schema.keys())
        if field_name in existing_indexes:
            logger.info(
                f"Index on {field_name} already exists in collection '{collection_name}'."
            )

            return

        await self.qadrant_async_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type,
        )

    # TODO: score <= 1.0
    async def dense_search_weighted(
        self,
        collection_name: str,
        query: str,
        weighted_metadata_items: list[WeightedMetadataItem],
        k: int = 10,
        pre_k: int = 50,
        score_threshold: float | None = None,
        search_filter: models.Filter | None = None,
    ) -> list[RetrieverItem]:
        vector = await self.dense_embeddings.aembed_query(text=query)
        mult_expressions = [
            models.MultExpression(
                mult=[
                    wmi.weight,
                    models.FieldCondition(
                        key=wmi.key,
                        match=models.MatchValue(value=wmi.value),
                    ),
                ]
            )
            for wmi in weighted_metadata_items
        ]

        formula = models.MultExpression(
            mult=[
                "$score",
                models.SumExpression(sum=[1.0] + mult_expressions),
            ]
        )

        response = await self.qadrant_async_client.query_points(
            collection_name=collection_name,
            prefetch=models.Prefetch(
                query=vector,
                using="dense",
                limit=pre_k,
                filter=search_filter,
            ),
            query=models.FormulaQuery(formula=formula),
            limit=k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            RetrieverItem(
                text=p.payload["page_content"],  # type: ignore
                metadata=p.payload["metadata"],  # type: ignore
                score=p.score,
            )
            for p in response.points
        ]
