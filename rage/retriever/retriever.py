import os

from uuid import uuid4
from typing import Literal
from functools import lru_cache
from common.logger import get_logger

from pydantic import BaseModel, StrictStr, NonNegativeFloat

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from rage.meta.interfaces import TextChunk


QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_GRPC_PORT = os.getenv("QDRANT_GRPC_PORT")

DENSE_EMBED_DOC_CACHE_PATH = os.getenv("DENSE_EMBED_DOC_CACHE_PATH")
DENSE_EMBED_QUERY_CACHE_PATH = os.getenv("DENSE_EMBED_QUERY_CACHE_PATH")
FAST_EMBED_SPARSE_CACHE = os.getenv("FAST_EMBED_SPARSE_CACHE")


logger = get_logger(__name__)


class RetrieverItem(BaseModel):
    text: StrictStr
    metadata: dict
    score: NonNegativeFloat | None = None


class Retriever:
    def __init__(
        self,
        dense_embed_model_name: str = "text-embedding-3-large",
        dense_embed_dimensions: int = 256,
        dense_embed_chunk_size: int = 1024,
        dense_embed_show_progress_bar: bool = False,
        dense_embed_doc_cache_path: str | None = DENSE_EMBED_DOC_CACHE_PATH,
        dense_embed_query_cache_path: str | None = DENSE_EMBED_QUERY_CACHE_PATH,
        sparse_embed_model_name: str = "Qdrant/bm25",
    ):
        self.dense_embed_dimensions = dense_embed_dimensions
        self.dense_embeddings = self._get_dense_embeddings(
            model_name=dense_embed_model_name,
            dimensions=dense_embed_dimensions,
            chunk_size=dense_embed_chunk_size,
            show_progress_bar=dense_embed_show_progress_bar,
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

        self.search_type_map = {
            "dense": self._get_dense_vector_store,
            "hybrid": self._get_hybrid_vector_store,
        }

    def _get_dense_embeddings(
        self,
        model_name: str,
        dimensions: int,
        chunk_size: int,
        show_progress_bar: bool,
        dense_embed_doc_cache_path: str | None,
        dense_embed_query_cache_path: str | None,
    ) -> Embeddings:
        underlying_embeddings = OpenAIEmbeddings(
            model=model_name,
            dimensions=dimensions,
            show_progress_bar=show_progress_bar,
            chunk_size=chunk_size,
        )

        if dense_embed_doc_cache_path is None:
            return underlying_embeddings

        query_embedding_cache = (
            True
            if dense_embed_query_cache_path is None
            else LocalFileStore(root_path=dense_embed_query_cache_path)
        )

        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=LocalFileStore(
                root_path=dense_embed_doc_cache_path
            ),
            namespace=model_name,
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

    def create_collection(self, collection_name: str) -> None:
        if self.qadrant_client.collection_exists(
            collection_name=collection_name
        ):
            logger.warning(f"collection {collection_name} already exists.")
            return

        self.qadrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_embed_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    def insert_text_chunks(
        self,
        collection_name: str,
        text_chunks: list[TextChunk],
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
        vector_store.add_documents(documents=lg_documents, ids=uuids)

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
    ) -> list[RetrieverItem]:
        vector_store = self._get_dense_vector_store(
            collection_name=collection_name
        )

        results = await vector_store.asimilarity_search_with_score(
            query=query,
            k=k,
        )

        return self._parse_results(results=results)

    async def hybrid_search(
        self,
        collection_name: str,
        query: str,
        k: int = 10,
    ) -> list[RetrieverItem]:
        vector_store = self._get_hybrid_vector_store(
            collection_name=collection_name
        )

        results = await vector_store.asimilarity_search_with_score(
            query=query,
            k=k,
        )

        return self._parse_results(results=results)

    @lru_cache()
    def _get_retriever(
        self,
        collection_name: str,
        search_type: str,
        k: int,
    ) -> VectorStoreRetriever:
        vector_store = self.search_type_map[search_type](
            collection_name=collection_name
        )

        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
            },
        )

    async def retrieve(
        self,
        collection_name: str,
        query: str,
        k: int = 10,
        search_type: Literal["dense", "hybrid"] = "dense",
    ) -> list[RetrieverItem]:
        retriever = self._get_retriever(
            collection_name=collection_name,
            search_type=search_type,
            k=k,
        )

        results = await retriever.ainvoke(input=query)
        return [
            RetrieverItem(
                text=r.page_content,
                metadata=r.metadata,
            )
            for r in results
        ]
