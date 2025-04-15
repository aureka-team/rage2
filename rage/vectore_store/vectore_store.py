import os

from uuid import uuid4
from functools import lru_cache
from common.logger import get_logger

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from rage.meta.interfaces import TextChunk


QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_GRPC_PORT = os.getenv("QDRANT_GRPC_PORT")


logger = get_logger(__name__)


class VectoreStore:
    def __init__(
        self,
        embeddings_model_name: str = "text-embedding-3-large",
        embeddings_dimensions: int = 256,
        embeddings_chunk_size: int = 1024,
        sparse_embeddings_model_name: str = "Qdrant/bm25",
    ):
        self.embeddings_dimensions = embeddings_dimensions
        self.embeddings = OpenAIEmbeddings(
            model=embeddings_model_name,
            dimensions=self.embeddings_dimensions,
            show_progress_bar=True,
            chunk_size=embeddings_chunk_size,
        )

        self.sparse_embeddings = FastEmbedSparse(
            model_name=sparse_embeddings_model_name
        )

        self.qadrant_client = QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
        )

    @lru_cache()
    def _get_qdrant_vector_store(
        self,
        collection_name: str,
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qadrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
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
                    size=self.embeddings_dimensions,
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

        vector_store = self._get_qdrant_vector_store(
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

    async def search(
        self, collection_name: str, query: str, k: int = 10
    ) -> None:
        vector_store = self._get_qdrant_vector_store(
            collection_name=collection_name
        )

        results = await vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

        return results
