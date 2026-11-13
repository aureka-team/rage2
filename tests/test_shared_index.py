import base64
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient, models

from rage.api.routers.collection.create import create_collection as indexing
from rage.api.routers.collection.get.get_collection import collection_get_router
from rage.api.routers.collection.get.get_collection_objects import (
    collection_get_objects_router,
)
from rage.api.routers.collection.list.list_collections import collection_list_router
from rage.api.routers.collection.remove.remove_collection import (
    collection_remove_router,
)
from rage.api.routers.retriever.retrieve import retrieve_router
from rage.api.shared_index import CONTENT_COLLECTION
from rage.api.utils import get_retriever
from rage.meta.interfaces import Document, TextChunk
from rage.retriever import Retriever, RetrieverItem


class LocalRetriever(Retriever):
    """Use real Qdrant filters with deterministic vectors and no external services."""

    def __init__(self):
        self.qadrant_async_client = AsyncQdrantClient(":memory:")
        self.dense_embed_dimensions = 2
        self.searches = 0
        self.fail_insert = False

    async def insert_text_chunks(self, collection_name, text_chunks):
        await self.qadrant_async_client.upsert(
            collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector={"dense": [1.0, 0.0]},
                    payload={"page_content": chunk.text, "metadata": chunk.metadata},
                )
                for chunk in text_chunks
            ],
        )
        if self.fail_insert:
            raise RuntimeError("embedding batch failed")

    async def dense_search(
        self, collection_name, query, k, score_threshold, search_filter
    ):
        self.searches += 1
        response = await self.qadrant_async_client.query_points(
            collection_name,
            query=[1.0, 0.0],
            using="dense",
            query_filter=search_filter,
            limit=k,
        )
        return [
            RetrieverItem(
                text=point.payload["page_content"],
                metadata=point.payload["metadata"],
                score=point.score,
            )
            for point in response.points
        ]


@pytest.mark.asyncio
async def test_shared_index_enforces_resource_scope_and_preserves_published_content(
    monkeypatch,
):
    async def load_file(file, directory):
        data = base64.b64decode(file.base64_file)
        return [
            Document(text=data.decode(), metadata={"document_name": file.file_name})
        ], file.base64_file

    class Splitter:
        def split_documents(self, documents):
            return [TextChunk(**doc.model_dump(), num_tokens=1) for doc in documents]

    monkeypatch.setattr(indexing, "_load_file", load_file)
    monkeypatch.setattr(indexing, "MarkdownSplitter", Splitter)
    retriever = LocalRetriever()
    await retriever.create_collection("legacy")
    app = FastAPI()
    async def get_local_retriever():
        return retriever

    app.dependency_overrides[get_retriever] = get_local_retriever
    for router in (
        indexing.collection_create_router,
        collection_get_router,
        collection_get_objects_router,
        collection_list_router,
        collection_remove_router,
        retrieve_router,
    ):
        app.include_router(router)

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:

        async def index(name, text):
            return await client.post(
                "/rage/collection/create",
                json={
                    "name": name,
                    "language": "en",
                    "overwrite": True,
                    "collection_files": [
                        {
                            "file_name": name,
                            "file_type": "text/plain",
                            "base64_file": base64.b64encode(text.encode()).decode(),
                        }
                    ],
                },
            )

        async def search(names):
            response = await client.post(
                "/rage/retriever/retrieve",
                json={
                    "collection_names": names,
                    "query_text": "query",
                    "search_mode": "dense",
                },
            )
            response.raise_for_status()
            return response.json()

        assert (await index("public", "original")).status_code == 200
        assert (await index("private", "private text")).status_code == 200
        assert (await client.post("/rage/collection/list")).json() == [
            "private",
            "public",
        ]

        result = await search(["public", "missing", "legacy"])
        assert result["invalid_collections"] == ["missing", "legacy"]
        assert result["error"] is True
        assert [item["text"] for item in result["retriever_items"]] == ["original"]
        assert result["retriever_items"][0]["collection_metadata"] == {"name": "public"}
        response = await client.post(
            "/rage/collection/get", json={"collection_name": "public"}
        )
        assert response.json()["collection_metadata"] == {
            "name": "public",
            "language": "en",
            "documents": ["public"],
            "files_md5": ["b3JpZ2luYWw="],
        }
        assert not (await search([]))["retriever_items"]
        assert retriever.searches == 1
        assert len((await search(["public", "private"]))["retriever_items"]) == 2
        assert retriever.searches == 2

        retriever.fail_insert = True
        assert (await index("public", "partial")).status_code == 500
        assert [
            item["text"] for item in (await search(["public"]))["retriever_items"]
        ] == ["original"]
        retriever.fail_insert = False
        assert (await index("public", "replacement")).status_code == 200
        objects = await client.post(
            "/rage/collection/get_objects",
            json={"collection_name": "public", "properties": ["text"], "filters": []},
        )
        assert [obj["properties"]["text"] for obj in objects.json()] == ["replacement"]
        assert (
            await retriever.qadrant_async_client.count(CONTENT_COLLECTION)
        ).count == 2

        response = await client.post("/rage/collection/remove", json={"name": "public"})
        assert response.json() == {"removed": True}
        result = await search(["public", "private"])
        assert [item["text"] for item in result["retriever_items"]] == ["private text"]
        assert (
            await retriever.qadrant_async_client.count(CONTENT_COLLECTION)
        ).count == 1

    await retriever.qadrant_async_client.close()
